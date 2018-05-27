#include <cuda.h>

extern "C" {

    __global__ void saxpy(float* res, float* x, float* y, int N) {
        int i = threadIdx.x;
        if (i >= N) return;
        res[i] = x[i] + y[i];
    }

    __global__ void relu(float* x, int N) {
        int i = threadIdx.x;
        if (i >= N) return;
        if (x[i] < 0) x[i] = 0;
    }

    __global__ void reluGrad(float* gx, float* gy, float* x, int N) {
        int i = threadIdx.x;
        if (i >= N) return;
        gx[i] = (x[i] <= 0) ? 0 : gy[i];
        if (x[i] < 0) x[i] = 0;
    }

    __global__ void sum(float* x, float* result, int N) {
        if (threadIdx.x == 0) return;
        result[0] = 0;
        for (int n = 0; n < N; ++n) {
            result[0] += x[n];
        }
    }

}