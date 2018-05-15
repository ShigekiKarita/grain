@compute(CompileFor.deviceOnly) module kernel;
import ldc.dcompute : GlobalPointer, kernel, compute, CompileFor;
import dcompute.std.index;
import dcompute.std.sync;

nothrow @nogc:

pragma(LDC_intrinsic, "llvm.nvvm.read.ptx.sreg.tid.x")
uint tid_x();

pragma(LDC_intrinsic, "llvm.log2")
float _log2(float);


extern (C)
@kernel void saxpy(GlobalPointer!(float) res,
                   GlobalPointer!(float) x,
                   GlobalPointer!(float) y,
                   size_t N)
{
    auto i = GlobalIndex.x;
    if (i >= N) return;
    res[i] = x[i] + y[i];
}

extern (C)
@kernel void relu(GlobalPointer!(float) x, size_t N)
{
    auto i = GlobalIndex.x;
    if (i >= N) return;
    if (x[i] < 0) x[i] = 0;
}

extern (C)
@kernel void reluGrad(GlobalPointer!float gx, GlobalPointer!float gy,
                      GlobalPointer!float x, size_t N)
{
    auto i = GlobalIndex.x;
    if (i >= N) return;
    gx[i] = (x[i] < 0) ? 0 : gy[i];
}

version = naive;

// http://www.toffee.jp/streaming/gpgpu/gpgpu_programming/2015/gpgpu_programming07.pdf

extern (C)
@kernel void sum(GlobalPointer!float x, GlobalPointer!float result, int N) {

    version (naive) {
        if (GlobalIndex.x != 0) return;
        result[0] = 0;
        for (int n = 0; n < N; ++n) {
            result[0] += x[n];
        }
    }

    version (reduce) {
    // import core.stdc.math;
    auto tx = SharedDimension.x;
    auto i = GlobalIndex.x * GlobalDimension.x + tx;
    int stride = 1;
    const nstep = _log2(N);
    for (int step = 1; step <= nstep; ++step) {
        if (tx % (2 * stride)== 0) {
            x[i] += x[i + stride];
        }
        barrier(); // eq to __syncthreads();
    }
    }
}
