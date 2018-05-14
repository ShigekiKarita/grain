#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128 

#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <iostream>

// C-style indexing
int ci(int row, int column, int nColumns) {
  return row*nColumns+column;
}
 
int main(void)
{
  int rowD = 5; // number of rows of D
  int colD = 8; // number of columns of D
  int rowE = colD; // number of rows of E
  int colE = 2; // number of columns of E
  int rowF = rowD;
  int colF = colE;

  // initialize data
  thrust::device_vector<float> D(rowD * colD);
  thrust::device_vector<float> E(rowE * colE);
  thrust::device_vector<float> F(rowF * colF);
  for (size_t i = 0; i < rowD; i++){
    for (size_t j = 0; j < colD; j++){
      D[ci(i,j,colD)]=i+j;
      std::cout << D[ci(i,j,colD)] << " ";
    }
    std::cout << "\n";
  }

  for (size_t i = 0; i < rowE; i++){
    for (size_t j = 0; j < colE; j++){
      E[ci(i,j,colE)]=i+j;
      std::cout << E[ci(i,j,colE)] << " ";
    }
    std::cout << "\n";
  }

  for (size_t i = 0; i < rowF; i++)
    for (size_t j = 0; j < colF; j++)
      F[ci(i,j,colF)]=0;

  cublasHandle_t handle;

    /* Initialize CUBLAS */
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "!!!! CUBLAS initialization error\n";
  }
  
  float alpha = 1.0f;float beta = 0.0f;
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                      colE, rowD, colD, 
                                      &alpha, thrust::raw_pointer_cast(&E[0]), colE, 
                                              thrust::raw_pointer_cast(&D[0]), colD, 
                                      &beta,  thrust::raw_pointer_cast(&F[0]), colE);// colE  x rowD
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! kernel execution error.\n";
  }

  for (size_t i = 0; i < rowF; i++){
    for (size_t j = 0; j < colF; j++){
      std::cout << F[ci(i,j,colF)] << " ";
    }
    std::cout << "\n";
  }

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "!!!! shutdown error (A)\n";
  }


  return 0;
}
