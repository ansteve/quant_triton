#!/usr/bin/python3

import torch
import triton

import my_marlin_cuda
import time

from dequant import *


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

    a = make_tensor(m, k, dtype=torch.float16)
    b = make_tensor(k//8, n, dtype=torch.int32)
    c = torch.zeros((m, n), dtype=torch.half, device="cuda")
    g_idx = torch.tensor([i // group_size for i in range(k)], dtype=torch.int32, device="cuda")
    workspace = torch.zeros(n//128*16, device="cuda")

    zeros = make_tensor(g, n//8, torch.int32)
    scales = make_tensor(g, n, torch.float16)
    num_groups = scales.shape[0]
    outfeatures = scales.shape[1]

    numels = c.numel()
    maxq = 2**nbits - 1
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)  # noqa: E731

    # Warmup
    for i in range(2):
        my_marlin_cuda.mul(a, b, c, scales, workspace, -1, -1, -1, 16)
        print(c)
        dequant_kernel_248[grid](
            g_idx,
            scales,
            b,
            zeros,
            c,
            numels,
            maxq=maxq,
            bits=nbits,
            outfeatures=outfeatures,
            num_groups=num_groups,
            X_BLOCK=1024,
        )
        print(c)
        # assert torch.allclose(c, output_split_k)

    # Measure marlin time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(10):
        my_marlin_cuda.mul(a, b, c, scales, workspace, -1, -1, -1, 16)
        # marlin.mul(a, b, c, scales, workspace, sms=108)
    torch.cuda.synchronize()
    end_time = time.time()
    print("Marlin time:", end_time - start_time)

    # Measure triton time
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(10):
        dequant_kernel_248[grid](
            g_idx,
            scales,
            b,
            zeros,
            c,
            numels,
            maxq=maxq,
            bits=nbits,
            outfeatures=outfeatures,
            num_groups=num_groups,
            X_BLOCK=1024,
        )
    torch.cuda.synchronize()
    end_time = time.time()
    print("dequant_kernel_248 time:", end_time - start_time)
