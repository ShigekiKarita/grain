extern "C"
__global__ void sum_naive(float* x, float* result, int N) {
    if (threadIdx.x == 0) return;
    result[0] = 0;
    for (int n = 0; n < N; ++n) {
        result[0] += x[n];
    }
}
