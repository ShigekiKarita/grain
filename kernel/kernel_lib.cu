#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/system/cuda/execution_policy.h>

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

GRAIN_GLOBAL void nllGrad(float* glogP, float coeff, const int* targetId, int ignoreIndex, uint batchSize) {
    GRAIN_PARALLEL_FOR(i, batchSize) {
        auto t = targetId[i];
        if (t != ignoreIndex) {
            glogP[i * batchSize + t] = coeff;
        }
    }
}

GRAIN_GLOBAL void addBias(float* y, const float* b, uint blen, uint ylen) {
    GRAIN_PARALLEL_FOR(i, ylen) {
        y[i] += b[i % blen];
    }
}


// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
  T C; // number of columns
  
  __host__ __device__
  linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i)
  {
    return i / C;
  }
};

// convert a linear index to a row index
template <typename T>
struct linear_index_to_col_index : public thrust::unary_function<T,T>
{
  T C; // number of columns
  
  __host__ __device__
  linear_index_to_col_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i)
  {
    return i % C;
  }
};


GRAIN_GLOBAL void addBiasGrad(const float* gy, float* gb, uint blen, uint ylen) {
    // compute row sums by summing values with equal row indices
    GRAIN_PARALLEL_FOR(i, ylen) {
        atomicAdd(gb + (i % blen), gy[i]);
    }

    // TODO use thrust
    // using I = uint;
    // auto key_iter = thrust::make_transform_iterator(thrust::counting_iterator<I>(0), linear_index_to_col_index<I>(blen));
    // thrust::reduce_by_key
    //     (thrust::cuda::par,
    //      key_iter, // keys_first
    //      key_iter + ylen, // keys_last
    //      thrust::device_ptr<const float>(gy), // values_first
    //      thrust::make_discard_iterator(), // keys_output
    //      thrust::device_ptr<float>(gb), // values_output
    //      thrust::equal_to<I>(), // binary_pred
    //      thrust::plus<float>()); // binary_o
}
