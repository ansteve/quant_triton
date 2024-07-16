#!/usr/bin/python3

import torch
import triton
from triton import language as tl
import itertools

import marlin 
import torch.nn as nn
# from auto_gptq.modeling._utils import autogptq_post_init
from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
import time

def make_dequant_configs(block_sizes, num_warps):
    configs = []
    for bs, ws in itertools.product(block_sizes, num_warps):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws))
    return configs


DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([128, 256, 512, 1024], [4, 8])


@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=["numels"])
@triton.jit
def dequant_kernel_248(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    numels,
    maxq: tl.constexpr,
    bits: tl.constexpr,
    outfeatures: tl.constexpr,
    num_groups: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    # Block indexing
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // outfeatures
    col_idx = x_index % outfeatures

    elements_per_feature: tl.constexpr = 32 // bits

    # Load parameters
    g_idx = tl.load(g_idx_ptr + (row_idx), None, eviction_policy="evict_last")
    qweights = tl.load(
        qweight_ptr + (col_idx + (outfeatures * (row_idx // elements_per_feature))),
        None,
    )

    wf_weights = (row_idx % elements_per_feature) * bits

    wf_zeros = (col_idx % elements_per_feature) * bits

    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    tl.device_assert(g_idx >= 0, "index out of bounds: 0 <= tmp0 < 0")
    groups = tl.where(tmp2, tmp1, g_idx)  # tmp3 are g_idx

    scales = tl.load(scales_ptr + (col_idx + (outfeatures * groups)), None).to(
        tl.float32
    )

    # Unpack weights
    weights = qweights >> wf_weights  # bit shift qweight

    weights = weights & maxq

    # Unpack zeros
    qzero_ncols: tl.constexpr = outfeatures // elements_per_feature
    qzeros = tl.load(
        qzeros_ptr + ((qzero_ncols * groups) + (col_idx // elements_per_feature)),
        None,
        eviction_policy="evict_last",
    )
    zeros = qzeros >> wf_zeros
    zeros = zeros & maxq

    # Dequantize
    zeros = zeros + 1
    weights = weights - zeros
    weights = weights.to(tl.float32)
    weights = scales * weights

    tl.store(out_ptr + (x_index), weights, mask=xmask)

def dequant248(qweight, scales, qzeros, g_idx, bits, maxq=None):
    """
    Launcher for triton dequant kernel.  Only valid for bits = 2, 4, 8
    """

    num_groups = scales.shape[0]
    outfeatures = scales.shape[1]
    infeatures = g_idx.shape[0]

    out = torch.empty((infeatures, outfeatures), device="cuda", dtype=torch.float16)
    numels = out.numel()
    maxq = 2**bits - 1 if maxq is None else maxq
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)  # noqa: E731

    dequant_kernel_248[grid](
        g_idx,
        scales,
        qweight,
        qzeros,
        out,
        numels,
        maxq=maxq,
        bits=bits,
        outfeatures=outfeatures,
        num_groups=num_groups,
    )
    return out

def quant_matmul_248(
    input, qweight, scales, qzeros, g_idx, bits, maxq=None, transpose=False
):
    W = dequant248(qweight, scales, qzeros, g_idx, bits, maxq=maxq)
    if transpose:
        return input @ W.t()
    return input @ W

def make_tensor(M, N, dtype):
    if dtype == torch.int32:
        # Fill with random integers for int32 type
        res = torch.randint(low=-2**31, high=2**31, size=(M, N), dtype=dtype, device="cuda")
    else:
        # Fill with normally distributed random values for other types
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    return res


if __name__ == '__main__':

    m = 16
    k = 4096
    n = 4096
    groupsize = 128
    g = k // groupsize

    nbits = 4
    group_size=128
    disable_exllama=True
    disable_exllamav2=False
    use_triton = False

    a = make_tensor(m, k, dtype=torch.float16)
    b = make_tensor(k//8, n, dtype=torch.int32)
    c = torch.zeros((m, n), dtype=torch.half, device="cuda")
    g_idx = torch.tensor([i // group_size for i in range(k)], dtype=torch.int32, device="cuda")
    workspace = torch.zeros(n//128*16, device="cuda")

    # zeros = make_tensor(g, n//8, torch.int32)
    zeros = torch.zeros((g, n//8), dtype=torch.int32, device="cuda")
    scales = make_tensor(g, n, torch.float16)

    # Marlin
    # m, n, k = 16, 4096, 4096
    # A = torch.randn((m, k), dtype=torch.half, device="cuda")
    # B_ref, B, s = gen_quant4(k, n)
    # C = torch.zeros((m, n), dtype=torch.half, device="cuda")
    # workspace = torch.zeros(n // 128*16, device="cuda")

    # marlin.mul(a, b, c, scales, workspace, sms=108)
    output_split_k = quant_matmul_248(a, b, scales, zeros, g_idx, bits=nbits)
    # assert torch.allclose(c, output_split_k)
    

    # linear_class = dynamically_import_QuantLinear(
    # disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2,
    # use_triton=use_triton, desc_act=False, group_size=group_size, bits=nbits)
    linear_class = dynamically_import_QuantLinear(
        disable_exllama=False, disable_exllamav2=True,
        use_triton=False, use_tritonv2=False, use_marlin=False, desc_act=False, group_size=group_size, bits=nbits)

    linear = linear_class(
        bits=nbits,
        group_size=group_size,
        infeatures=k,
        outfeatures=n,
        bias=0,
    )

    device = torch.device('cuda')

    # linear.qweight = torch.randint(-100, 100, size=linear.qweight.shape, dtype=torch.int32)
    # linear.scales = linear.scales + 0.002
    linear.qweight = b
    linear.scales = scales
    linear.qzeros = zeros
    # linear.B = b
    # linear.s = scales

    linear = linear.eval().to(device)
    # linear = autogptq_post_init(linear, use_act_order=False)

    b_fake = torch.randn((k, n), dtype=torch.float16, device="cuda")

    # Warmup
    for i in range(3):
        c = linear(a)
        # marlin.mul(a, b, c, scales, workspace, sms=108)
        output_split_k = quant_matmul_248(a, b, scales, zeros, g_idx, bits=nbits)
        print(c)
        print(output_split_k)
        # assert torch.allclose(c, output_split_k)


    # Measure linear time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(7):
        linear(a)
        # marlin.mul(a, b, c, scales, workspace, sms=108)
    torch.cuda.synchronize()
    end_time = time.time()
    print("linear time:", end_time - start_time)

    # Measure matmul_split_k time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(7):
        quant_matmul_248(a, b, scales, zeros, g_idx, bits=nbits)
    torch.cuda.synchronize()
    end_time = time.time()
    print("matmul_split_k time:", end_time - start_time)

