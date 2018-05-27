#include <cuda.h>


#ifdef __CUDACC__
#    define GRAIN_DEVICE_HOST __device__ __host__
#    define GRAIN_GLOBAL __global__
#    define GRAIN_PARALLEL_FOR(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#else
#    define GRAIN_DEVICE_HOST
#    define GRAIN_GLOBAL
#    define GRAIN_PARALLEL_FOR(i, n) \
    _Pragma("omp parallel for") \
    for (int i = 0; i < (n); ++i)
#endif


// template <typename T>
// GRAIN_GLOBAL void saxpy(T* res, const T* x, const T* y, int n) {
GRAIN_GLOBAL void saxpy(float* res, const float* x, const float* y, int n) {
    GRAIN_PARALLEL_FOR(i, n) {
        res[i] = x[i] + y[i];
    }
}

GRAIN_GLOBAL void relu(float* x, int n) {
    GRAIN_PARALLEL_FOR(i, n) {
        if (x[i] < 0) x[i] = 0;
    }
}

GRAIN_GLOBAL void reluGrad(float* gx, const float* gy, const float* x, int n) {
    GRAIN_PARALLEL_FOR(i, n) {
        gx[i] = (x[i] <= 0) ? 0 : gy[i];
    }
}

GRAIN_GLOBAL void sum(const float* x, float* result, int N) {
    if (threadIdx.x != 0) return;
    result[0] = 0;
    for (int n = 0; n < N; ++n) {
        result[0] += x[n];
    }
}

GRAIN_GLOBAL void nll(float* loss, uint* count, const float* logp, const int* targetId, int ignoreIndex, uint batchSize) {
    GRAIN_PARALLEL_FOR(i, batchSize) {
        auto t = targetId[i];
        if (t != ignoreIndex) {
            atomicAdd(loss, -logp[i * batchSize + t]);
            atomicAdd(count, 1);
        }
    }
}

