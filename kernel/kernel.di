// D import file generated from 'kernel/kernel.d'
// @(compute(CompileFor.deviceOnly))
module kernel;
// import ldc.dcompute : GlobalPointer, kernel, compute, CompileFor;
// import dcompute.std.index;
// import dcompute.std.atomic;
// import dcompute.std.sync;
nothrow @nogc extern (C++) 
{
	/* @(kernel) */void saxpy(float* res, const float* x, const float* y, int N);
	/* @(kernel) */void relu(float* x, int N);
	/* @(kernel) */void reluGrad(float* gx, const float* gy, const float* x, int N);
	/* @(kernel) */void sum(const float* x, float* result, int N);
    /* @(kernel) */void sum_faster(const float* x, float* result, uint n, uint N);
	/* @(kernel) */void nll(float* loss, uint* count, const float* logp, const int* targetId, int ignoreIndex, uint batchSize, int logpStride);
	/* @(kernel) */void nllGrad(float* glogP, float coeff, const int* targetId, int ignoreIndex, uint batchSize, int logpStride);
	/* @(kernel) */void addBias(float* y, const float* b, uint blen, uint ylen);
	/* @(kernel) */void addBiasGrad(const float* gy, float* gb, uint blen, uint ylen);
	/* @(kernel) */void reciprocal(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void log(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void log2(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void log10(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void exp(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void exp2(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void exp10(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void sin(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void cos(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void tan(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void pow(float power, float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void powGrad(float power, float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void neg(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void abs(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void absGrad(float* x, uint len, uint ndim, const uint* shape, const uint* strides);
	/* @(kernel) */void embedding(const float* w, const int* x, float* y, uint nvocab, uint nembed, uint nbatch);
	/* @(kernel) */void embeddingGrad(float* gw, const int* x, const float* gy, uint nvocab, uint nembed, uint nbatch);
}
