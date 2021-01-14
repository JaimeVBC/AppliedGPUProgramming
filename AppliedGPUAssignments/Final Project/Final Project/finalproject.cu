#include <stdio.h>
#include <cuda_runtime_api.h>

#include "macros.h"
#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
#include <assert.h>

/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
 */

 /*
  * M = # of rows
  * N = # of columns
  */
int M = 1024;
int N = 1024;


/*
 * Multiply nxn (square) matrices function in the CPU
 */
void mult_matrices_CPU(int M, float * A, float * B, float* outC)
{
    int i, j, elemPtr;
    int contPos;
    float sumElem;
    for (contPos = 0; contPos < M*M; contPos++)
    {
        sumElem = 0;
        i = contPos / M;
        j = contPos % M;
        for (elemPtr = 0; elemPtr < M; elemPtr++)
        {
            sumElem += A[i * M + elemPtr] * B[j + M * elemPtr];
        }
        outC[contPos] = sumElem;
    }
    return;
}

void multiply(float* a, int row1, int col1, float* b, int row2, int col2, float * d)
{
    assert(col1 == row2);

    //int size = row1 * col2;


    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            float sum = 0;
            for (int k = 0; k < col1; k++)
                sum = sum + a[i * col1 + k] * b[k * col2 + j];
            d[i * col2 + j] = sum;
        }
    }

    /*for (int i = 0; i < size; i++) {
        if (i % col2 == 0) {
            printf("\n");
        }
        printf("%d ", d[i]);
    }*/

}


/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_dense_matrix(int M, int N, float** outA)
{
    int i, j;
    float rMax = (float)RAND_MAX;
    float* A = (float*)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            float* curr = A + (j * M + i);

            if (r % 3 > 0) // About 2/3 percent of elements would be zero. Module operation can be changed to variate the zero elements rate.
            {
                *curr = 0.0f;
            }
            else // About 1/3 of elements would be non-zero elements.
            {
                float dr = (float)r;
                *curr = (dr / rMax) * 100.0f;
            }

            if (*curr != 0.0f)
            {
                totalNnz++; // Total count of non-zero elements
            }
        }
    }

    *outA = A;
    return totalNnz;
}

/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_sparse_matrix(int M, int N, float** outA)
{
    int i, j;
    float rMax = (float)RAND_MAX;
    float* A = (float*)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            float* curr = A + (j * M + i);

            if (r % 20 > 0) // About 95% percent of elements would be zero. Module operation can be changed to variate the zero elements rate.
            {
                *curr = 0.0f;
            }
            else // About a 5% of elements would be non-zero elements.
            {
                float dr = (float)r;
                *curr = (dr / rMax) * 100.0f;
            }

            if (*curr != 0.0f)
            {
                totalNnz++; // Total count of non-zero elements
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
    float* C, * dC, *C_check;

    int* dANnzPerRow; //Number of elements per row that are not zero
    float* dCsrValA; // Vector with non-zero elements
    int* dCsrRowPtrA; // CSR rows pointer of matrix A
    int* dCsrColIndA; // CSR column indexes of matrix A
    int totalANnz = 0; // Total number of non-zero elements in matrix A

    float alpha = 1.0f; // Scalar Alpha
    float beta = 0.0f; // Scalar Beta

    float tol = 0.001;

    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t Adescr = NULL; // Generic descriptor of dense matrix A

    cusparseDnMatDescr_t ADndescr = NULL; // Dense descriptor of dense matrix B
    cusparseDnMatDescr_t Bdescr = NULL; // Descriptor of dense matrix B
    cusparseDnMatDescr_t Cdescr = NULL; // Descriptor of dense matrix C

    cusparseSpMatDescr_t ASpdescr = NULL; // Descriptor of sparse version of dense matrix A

    //Create buffer in the device (necessary to perform function cusparseSpMM)
    void* dbuffer = NULL;
    //Create bufferSize
    size_t buffersize = 0;

    // Create the cuSPARSE handle
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Generation of random dense matrices A and B. (Memory allocation included)
    srand(9384);
    int trueANnz = generate_random_sparse_matrix(M, N, &A);
    int trueBNnz = generate_random_dense_matrix(N, M, &B);
    // Allocation of enough memory to store C values
    C = (float*)malloc(sizeof(float) * M * M);
    C_check = (float*)malloc(sizeof(float) * M * M);

    printf("A:\n");
    print_partial_matrix(A, M, N, 10, 10);
    printf("B:\n");
    print_partial_matrix(B, N, M, 10, 10);

    //Perform CPU mult of A and B
    multiply(A,M,M,B,M,M,C_check);

    // Allocate device memory for the dense form of the matrices A, B, C and the vector containing the number of non-zero elements per row of matrix A
    CHECK_CUDA(cudaMalloc((void**)&dA, sizeof(float) * M * N));
    CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(float) * N * M));
    CHECK_CUDA(cudaMalloc((void**)&dC, sizeof(float) * M * M));
    CHECK_CUDA(cudaMalloc((void**)&dANnzPerRow, sizeof(int) * M));

   
    // Construct a descriptor of the dense matrix A
    CHECK_CUSPARSE(cusparseCreateMatDescr( (cusparseMatDescr_t *)&Adescr));
    CHECK_CUSPARSE(cusparseSetMatType((cusparseMatDescr_t)Adescr, CUSPARSE_MATRIX_TYPE_GENERAL)); // redundant because it's the default value
    CHECK_CUSPARSE(cusparseSetMatIndexBase((cusparseMatDescr_t)Adescr, CUSPARSE_INDEX_BASE_ZERO)); // redundant because it's the default value*/
    
    // Construct a descriptor of the dense matrix B
    /*CHECK_CUSPARSE(cusparseCreateMatDescr( &Bdescr));
    CHECK_CUSPARSE(cusparseSetMatType( Bdescr, CUSPARSE_MATRIX_TYPE_GENERAL)); // redundant because it's the default value
    CHECK_CUSPARSE(cusparseSetMatIndexBase( Bdescr, CUSPARSE_INDEX_BASE_ZERO)); // redundant because it's the default value

    // Construct a descriptor of the dense matrix C
    CHECK_CUSPARSE(cusparseCreateMatDescr( &Cdescr));
    CHECK_CUSPARSE(cusparseSetMatType( Cdescr, CUSPARSE_MATRIX_TYPE_GENERAL)); // redundant because it's the default value
    CHECK_CUSPARSE(cusparseSetMatIndexBase( Cdescr, CUSPARSE_INDEX_BASE_ZERO)); // redundant because it's the default value
    */

    // Transfer the dense matrixes A, B and C to the device
    CHECK_CUDA(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, sizeof(float) * N * M, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * M * M));

    // Initialize the dense matrix A descriptor
    CHECK_CUSPARSE(cusparseCreateDnMat(&ADndescr, M, N, M, dA, CUDA_R_32F, CUSPARSE_ORDER_COL));
    
    // Initialize the dense matrix B descriptor
    CHECK_CUSPARSE(cusparseCreateDnMat(&Bdescr, M, N, M, dB, CUDA_R_32F, CUSPARSE_ORDER_COL));

    // Initialize the dense matrix C descriptor
    CHECK_CUSPARSE(cusparseCreateDnMat(&Cdescr, M, N, M, dC, CUDA_R_32F, CUSPARSE_ORDER_COL));


    // Compute the number of non-zero elements in A. Total ammount and array with NZ per row
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, Adescr,
        dA, M, dANnzPerRow, &totalANnz));

    if (totalANnz != trueANnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
            "value: expected %d but got %d\n", trueANnz, totalANnz);
        return 1;
    }

    // Allocate device memory to store the vectors related to the sparse CSR representation of A
    CHECK_CUDA(cudaMalloc((void**)&dCsrValA, sizeof(float) * totalANnz));
    CHECK_CUDA(cudaMalloc((void**)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    CHECK_CUDA(cudaMalloc((void**)&dCsrColIndA, sizeof(int) * totalANnz));
    
    // Initialize a sparse descriptor of the matrix A
    CHECK_CUSPARSE(cusparseCreateCsr(&ASpdescr, M, N, totalANnz, dCsrRowPtrA, dCsrColIndA, dCsrValA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));


    // Convert A from a dense formatting to a CSR formatting, using the GPU. Assignment of this CSR A matrix to the dA 
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle,(cusparseDnMatDescr_t)ADndescr,ASpdescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,&buffersize));
    CHECK_CUDA(cudaMalloc((void**)&dbuffer, buffersize));

    // Convert A from a dense formatting to a CSR formatting, using the GPU. Assignment of this CSR A matrix to the dA 
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle,(cusparseDnMatDescr_t)ADndescr,ASpdescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,dbuffer));
    
    // Convert A from a dense formatting to a CSR formatting, using the GPU. Assignment of this CSR A matrix to the dA 
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle,(cusparseDnMatDescr_t)ADndescr,ASpdescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,dbuffer));
    
    
    //Get the buffer size for the workspace
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,  ASpdescr, (cusparseDnMatDescr_t)Bdescr, &beta, (cusparseDnMatDescr_t)Cdescr, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &buffersize));

    
    // Perform matrix-matrix multiplication with the CSR-formatted matrix A
    CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, ASpdescr, (cusparseDnMatDescr_t)Bdescr, &beta, (cusparseDnMatDescr_t)Cdescr, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, dbuffer));
    
    // Copy the result vector back to the host
    CHECK_CUDA(cudaMemcpy(C, dC, sizeof(float) * M * M, cudaMemcpyDeviceToHost));

    printf("C:\n");
    print_partial_matrix(C, M, M, 10, 10);

    printf("C_check:\n");
    print_partial_matrix(C_check, M, M, 10, 10);

    bool check = true;
    for (int i = 0; i < M * M && check; i++)
    {
        if (abs(C[i] - C_check[i]) > tol)
        {
            check = false;
        }
    }
    if (check){ printf("CPU and GPU solutions ARE the same");  }
    else      { printf("CPU and GPU solutions ARE NOT the same");  }

    free(A);
    free(B);
    free(C);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dANnzPerRow));
    CHECK_CUDA(cudaFree(dCsrValA));
    CHECK_CUDA(cudaFree(dCsrRowPtrA));
    CHECK_CUDA(cudaFree(dCsrColIndA));

    CHECK_CUSPARSE(cusparseDestroySpMat(ASpdescr));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(Adescr));
    CHECK_CUSPARSE(cusparseDestroyDnMat(Bdescr));
    CHECK_CUSPARSE(cusparseDestroyDnMat(Cdescr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}
