/**
   cuBLAS wrappers
 */
module grain.cublas;
version (grain_cuda):

import derelict.cuda;

/++
 TODO: make derelict-cublas (already exist?)
 +/

extern (C):

alias cublasStatus_t = int;
///
enum : cublasStatus_t {
    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14,
    CUBLAS_STATUS_NOT_SUPPORTED   =15,
    CUBLAS_STATUS_LICENSE_ERROR   =16
}

///
struct cublasContext;
///
alias cublasHandle_t = cublasContext*;
///
alias cublasOperation_t = int;
///
enum : cublasOperation_t {
    CUBLAS_OP_N, // the non-transpose operation is selected
    CUBLAS_OP_T, // the transpose operation is selected
    CUBLAS_OP_C // the conjugate transpose operation is selected
}


// TODO: parse and retrieve cublas_api.h

///
cublasStatus_t cublasCreate_v2(cublasHandle_t*);
///
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

 cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);

/*
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc);
*/

///
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc);
///
cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n,
                              const float           *alpha,
                              const float           *x, int incx,
                              float                 *y, int incy);
///
cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n,
                              const double          *alpha,
                              const double          *x, int incx,
                              double                *y, int incy);

/// emit error message string from enum
auto cublasGetErrorEnum(cublasStatus_t error) {
    final switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
}
