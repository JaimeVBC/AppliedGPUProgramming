#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
#include <cuda.h>

/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
 */

 /*
  * M = # of rows
  * N = # of columns
  */
int M = 16;
int N = 16;

/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_dense_matrix(int M, int N, float** outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float* A = (float*)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            float* curr = A + (j * M + i);

            if (r % 3 > 0)
            {
                *curr = 0.0f;
            }
            else
            {
                double dr = (double)r;
                *curr = (dr / rMax) * 100.0;
            }

            if (*curr != 0.0f)
            {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

void print_partial_matrix(float* M, int nrows, int ncols, int max_row,
    int max_col)
{
    int row, col;

    for (row = 0; row < max_row; row++)
    {
        for (col = 0; col < max_col; col++)
        {
            printf("%2.2f ", M[row * ncols + col]);
        }
        printf("...\n");
    }
    printf("...\n");
}

int main(int argc, char** argv)
{
    float* A, * dA;
    float* B, * dB;
    float* C, * dC;

    int* dANnzPerRow; //Number of elements per row that are not zero
    float* dCsrValA; // Vector with non-zero elements
    int* dCsrRowPtrA; // por d�nde va el puntero del vector
    int* dCsrColIndA; //columna en la que est� cada elemento
    int totalANnz = 0; //valores no cero
    float alpha = 3.0f;
    float beta = 4.0f;
    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t Adescr = NULL;
    cusparseMatDescr_t Bdescr = NULL;
    cusparseMatDescr_t Cdescr = NULL;

    // Generate input
    srand(9384);
    int trueANnz = generate_random_dense_matrix(M, N, &A);
    int trueBNnz = generate_random_dense_matrix(N, M, &B);
    C = (float*)malloc(sizeof(float) * M * M);

    printf("A:\n");
    print_partial_matrix(A, M, N, 10, 10);
    printf("B:\n");
    print_partial_matrix(B, N, M, 10, 10);

    // Create the cuSPARSE handle
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Allocate device memory for vectors and the dense form of the matrix A
    CHECK(cudaMalloc((void**)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void**)&dB, sizeof(float) * N * M));
    CHECK(cudaMalloc((void**)&dC, sizeof(float) * M * M));
    CHECK(cudaMalloc((void**)&dANnzPerRow, sizeof(int) * M));

   
    // Construct a descriptor of the matrix A
    CHECK_CUSPARSE(cusparseCreateMatDescr( &Adescr));
    CHECK_CUSPARSE(cusparseSetMatType( Adescr, CUSPARSE_MATRIX_TYPE_GENERAL)); // redundant because it's the default value
    CHECK_CUSPARSE(cusparseSetMatIndexBase( Adescr, CUSPARSE_INDEX_BASE_ZERO)); // redundant because it's the default value

    // Construct a descriptor of the matrix B
    CHECK_CUSPARSE(cusparseCreateMatDescr( &Bdescr));
    CHECK_CUSPARSE(cusparseSetMatType( Bdescr, CUSPARSE_MATRIX_TYPE_GENERAL)); // redundant because it's the default value
    CHECK_CUSPARSE(cusparseSetMatIndexBase( Bdescr, CUSPARSE_INDEX_BASE_ZERO)); // redundant because it's the default value

    // Construct a descriptor of the matrix C
    CHECK_CUSPARSE(cusparseCreateMatDescr( &Cdescr));
    CHECK_CUSPARSE(cusparseSetMatType( Cdescr, CUSPARSE_MATRIX_TYPE_GENERAL)); // redundant because it's the default value
    CHECK_CUSPARSE(cusparseSetMatIndexBase( Cdescr, CUSPARSE_INDEX_BASE_ZERO)); // redundant because it's the default value

    // Initialize the dense matrix B descriptor
    CHECK_CUSPARSE(cusparseCreateDnMat((cusparseDnMatDescr_t*)&Bdescr, M, N, M, B, CUDA_R_32F, CUSPARSE_ORDER_COL));

    // Initialize the dense matrix C descriptor
    CHECK_CUSPARSE(cusparseCreateDnMat((cusparseDnMatDescr_t*)&Cdescr, M, N, M, C, CUDA_R_32F, CUSPARSE_ORDER_COL));

    // Transfer the input vectors and dense matrix A to the device
    CHECK(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, sizeof(float) * N * M, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0, sizeof(float) * M * M));

    // Compute the number of non-zero elements in A. Total ammount and array with NZ per row
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, Adescr,
        dA, M, dANnzPerRow, &totalANnz));

    if (totalANnz != trueANnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
            "value: expected %d but got %d\n", trueANnz, totalANnz);
        return 1;
    }

    // Allocate device memory to store the sparse CSR representation of A
    CHECK(cudaMalloc((void**)&dCsrValA, sizeof(float) * totalANnz));
    CHECK(cudaMalloc((void**)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    CHECK(cudaMalloc((void**)&dCsrColIndA, sizeof(int) * totalANnz));

    // Convert A from a dense formatting to a CSR formatting, using the GPU
    CHECK_CUSPARSE(cusparseSdense2csr(handle, M, N, (cusparseMatDescr_t) Adescr, dA, M, dANnzPerRow,
        dCsrValA, dCsrRowPtrA, dCsrColIndA));
     
    //Create bufferSize
    size_t buffersize;

    //Construct a sparse descriptor of the matrix A
    cusparseSpMatDescr_t ASpdescr = NULL;
    CHECK_CUSPARSE(cusparseCreateCsr(&ASpdescr, M, N, totalANnz, dCsrRowPtrA, dCsrColIndA, dCsrValA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    //Get the buffer size for the workspace
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,  ASpdescr, (cusparseDnMatDescr_t)Bdescr, &beta, (cusparseDnMatDescr_t)Cdescr, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &buffersize));

    //Create buffer
    void* buffer;
    cudaMalloc((void**)&buffer, buffersize);
    
    // Perform matrix-matrix multiplication with the CSR-formatted matrix A
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, ASpdescr, (cusparseDnMatDescr_t)Bdescr, &beta, (cusparseDnMatDescr_t)Cdescr, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, buffer));
    
    // Copy the result vector back to the host
    CHECK(cudaMemcpy(C, dC, sizeof(float) * M * M, cudaMemcpyDeviceToHost));

    printf("C:\n");
    print_partial_matrix(C, M, M, 10, 10);

    free(A);
    free(B);
    free(C);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaFree(dANnzPerRow));
    CHECK(cudaFree(dCsrValA));
    CHECK(cudaFree(dCsrRowPtrA));
    CHECK(cudaFree(dCsrColIndA));

    CHECK_CUSPARSE(cusparseDestroySpMat(ASpdescr));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(Adescr));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(Bdescr));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(Cdescr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}
