@compute(CompileFor.deviceOnly) module kernel;
import ldc.dcompute : GlobalPointer, kernel, compute, CompileFor;
import dcompute.std.index;

pure nothrow @nogc extern (C):

pragma(LDC_intrinsic, "llvm.nvvm.read.ptx.sreg.tid.x")
uint tid_x();

@kernel void saxpy(GlobalPointer!(float) res,
                   GlobalPointer!(float) x,
                   GlobalPointer!(float) y,
                   size_t N)
{
    auto i = GlobalIndex.x;
    if (i >= N) return;
    res[i] = x[i] + y[i];
}

@kernel void relu(GlobalPointer!(float) x, size_t N)
{
    auto i = GlobalIndex.x;
    if (i >= N) return;
    if (x[i] < 0) x[i] = 0;
}

@kernel void reluGrad(GlobalPointer!float gx, GlobalPointer!float gy,
                      GlobalPointer!float x, size_t N)
{
    auto i = GlobalIndex.x;
    if (i >= N) return;
    gx[i] = (x[i] < 0) ? 0 : gy[i];
}

