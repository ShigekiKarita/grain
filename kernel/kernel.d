@compute(CompileFor.deviceOnly) module kernel;
/++

NOTE: now every implementation is written in kernel_lib.cu. here are only definitions.

 +/

import ldc.dcompute : GlobalPointer, kernel, compute, CompileFor;
import dcompute.std.index;
import dcompute.std.atomic;
import dcompute.std.sync;

nothrow @nogc extern(C++):

// pragma(LDC_intrinsic, "llvm.log2")
// float _log2(float);

// pragma(LDC_intrinsic, "llvm.nvvm.atomic.load.add.f32.p1f32")
// float atomicAdd(out GlobalPointer!float, float);


// @kernel void saxpy(T)(T* res,
//                    const T* x,
//                    const T* y,
//                    int N);

@kernel void saxpy(float* res,
                   const float* x,
                   const float* y,
                   int N);
// {
//     auto i = GlobalIndex.x;
//     if (i >= N) return;
//     res[i] = x[i] + y[i];
// }

@kernel void relu(float* x, int N);
// {
//     auto i = GlobalIndex.x;
//     if (i >= N) return;
//     if (x[i] < 0) x[i] = 0;
// }

@kernel void reluGrad(float* gx, const float* gy,
                      const float* x, int N);
// {
//     auto i = GlobalIndex.x;
//     if (i >= N) return;
//     gx[i] = (x[i] <= 0) ? 0 : gy[i];
// }

// http://www.toffee.jp/streaming/gpgpu/gpgpu_programming/2015/gpgpu_programming07.pdf

@kernel void sum(const float* x, float* result, int N);

@kernel void nll(float* loss, uint* count, const float* logp, const int* targetId, int ignoreIndex, uint batchSize);

@kernel void nllGrad(float* glogP, float coeff, const int* targetId, int ignoreIndex, uint batchSize);


@kernel void addBias(float* y, const float* b, uint blen, uint ylen);

@kernel void addBiasGrad(const float* gy, float* gb, uint blen, uint ylen);
