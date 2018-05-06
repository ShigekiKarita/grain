import ldc.dcompute : GlobalPointer, kernel;

pure nothrow @nogc extern (C):

pragma(LDC_intrinsic, "llvm.nvvm.read.ptx.sreg.tid.x")
uint tid_x();

@kernel void saxpy(GlobalPointer!(float) res,
                   GlobalPointer!(float) x,
                   GlobalPointer!(float) y,
                   size_t N)
{
    auto i = tid_x(); // GlobalIndex.x;
    if (i >= N) return;
    res[i] = x[i] + y[i];
}
