@compute(CompileFor.deviceOnly) module kernel;
/++

NOTE: now every implementation is written in kernel_lib.cu. here are only definitions.

 +/

import ldc.dcompute : GlobalPointer, kernel, compute, CompileFor;
import dcompute.std.index;
import dcompute.std.atomic;
import dcompute.std.sync;

nothrow @nogc extern(C):

// pragma(LDC_intrinsic, "llvm.log2")
// float _log2(float);

// pragma(LDC_intrinsic, "llvm.nvvm.atomic.load.add.f32.p1f32")
// float atomicAdd(out GlobalPointer!float, float);

@kernel void saxpy(GlobalPointer!(float) res,
                   GlobalPointer!(float) x,
                   GlobalPointer!(float) y,
                   int N);
// {
//     auto i = GlobalIndex.x;
//     if (i >= N) return;
//     res[i] = x[i] + y[i];
// }

@kernel void relu(GlobalPointer!(float) x, size_t N);
// {
//     auto i = GlobalIndex.x;
//     if (i >= N) return;
//     if (x[i] < 0) x[i] = 0;
// }

@kernel void reluGrad(GlobalPointer!float gx, GlobalPointer!float gy,
                      GlobalPointer!float x, size_t N);
// {
//     auto i = GlobalIndex.x;
//     if (i >= N) return;
//     gx[i] = (x[i] <= 0) ? 0 : gy[i];
// }

// http://www.toffee.jp/streaming/gpgpu/gpgpu_programming/2015/gpgpu_programming07.pdf

@kernel void sum(GlobalPointer!float x, GlobalPointer!float result, int N);


@kernel void nll(GlobalPointer!float x, GlobalPointer!long t, out GlobalPointer!float loss, int ignoreIndex, int N);
