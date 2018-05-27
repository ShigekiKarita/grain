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



extern "C" {
    GRAIN_GLOBAL void saxpy(float* res, float* x, float* y, int n) {
        GRAIN_PARALLEL_FOR(i, n) {
            res[i] = x[i] + y[i];
        }
    }

    GRAIN_GLOBAL void relu(float* x, int n) {
        GRAIN_PARALLEL_FOR(i, n) {
            if (x[i] < 0) x[i] = 0;
        }
    }

    GRAIN_GLOBAL void reluGrad(float* gx, float* gy, float* x, int n) {
        GRAIN_PARALLEL_FOR(i, n) {
            gx[i] = (x[i] <= 0) ? 0 : gy[i];
        }
    }

    GRAIN_GLOBAL void sum(float* x, float* result, int N) {
        if (threadIdx.x != 0) return;
        result[0] = 0;
        for (int n = 0; n < N; ++n) {
            result[0] += x[n];
        }
    }

}