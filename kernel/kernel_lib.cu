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

GRAIN_GLOBAL void nll(float* loss, uint* count, const float* logp, const int* targetId, int ignoreIndex, uint batchSize, int classSize) {
    GRAIN_PARALLEL_FOR(i, batchSize) {
        auto t = targetId[i];
        if (t != ignoreIndex) {
            atomicAdd(loss, -logp[i * classSize + t]);
            atomicAdd(count, 1);
        }
    }
}

GRAIN_GLOBAL void nllGrad(float* glogP, float coeff, const int* targetId, int ignoreIndex, uint batchSize, int classSize) {
    GRAIN_PARALLEL_FOR(i, batchSize) {
        auto t = targetId[i];
        if (t != ignoreIndex) {
            glogP[i * classSize + t] = coeff;
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

// x[i0, i1, ..., iN-1] = x[i0 * strides[N-1] + i1 * strides[N-2] + ... + iN-1 * strides[0]]
__device__ uint indexof(uint i, uint ndim, const uint* shape, const uint* strides) {
    uint idx = i;
    uint pos = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        pos += (idx % shape[d]) * strides[d];
        idx /= shape[d];
    }
    return pos;
}

/// TODO generalize this nd map function with template
/// TODO define all math functions in CUDA
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix

#define GRAIN_ND_EACH(name, func)                                       \
    GRAIN_GLOBAL void name(float* x, uint len, uint ndim, const uint* shape, const uint* strides) { \
        uint idx;                                                       \
        GRAIN_PARALLEL_FOR(i, len) {                                    \
            idx = indexof(i, ndim, shape, strides);                     \
            x[idx] = func(x[idx]);                                      \
        }                                                               \
    }


// fast-math functions https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions
GRAIN_GLOBAL void reciprocal(float* x, uint len, uint ndim, const uint* shape, const uint* strides) {
    uint idx;
    GRAIN_PARALLEL_FOR(i, len) {
        idx = indexof(i, ndim, shape, strides);
        x[idx] = 1.0f / x[idx];
    }
}

GRAIN_ND_EACH(log, logf)
GRAIN_ND_EACH(log2, log2f)
GRAIN_ND_EACH(log10, log10f)

GRAIN_ND_EACH(exp, expf)
GRAIN_ND_EACH(exp2, exp2f)
GRAIN_ND_EACH(exp10, exp10f)

GRAIN_ND_EACH(cos, cosf)
GRAIN_ND_EACH(sin, sinf)
GRAIN_ND_EACH(tan, tanf)

GRAIN_GLOBAL void pow(float power, float* x, uint len, uint ndim, const uint* shape, const uint* strides) {
    uint idx;
    GRAIN_PARALLEL_FOR(i, len) {
        idx = indexof(i, ndim, shape, strides);
        x[idx] = powf(x[idx], power);
    }
}
