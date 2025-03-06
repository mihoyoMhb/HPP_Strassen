#include <stdlib.h>
#include <string.h>
#include "matrix.h"
// Serial Strassen's algorithm (recursive, without OpenMP tasks)
void strassen_serial(double *A, double *B, double *C, int n) {
    if (n <= BASE_SIZE) {
        multiply_standard_serial(A, B, C, n);
        return;
    }
    
    int newSize = n / 2;
    int size = newSize * newSize;
    
    // Allocate submatrices for A
    double *A11 = (double*)malloc(size * sizeof(double));
    double *A12 = (double*)malloc(size * sizeof(double));
    double *A21 = (double*)malloc(size * sizeof(double));
    double *A22 = (double*)malloc(size * sizeof(double));
    
    // Allocate submatrices for B
    double *B11 = (double*)malloc(size * sizeof(double));
    double *B12 = (double*)malloc(size * sizeof(double));
    double *B21 = (double*)malloc(size * sizeof(double));
    double *B22 = (double*)malloc(size * sizeof(double));
    
    // Split A and B into 4 submatrices each
    for (int i = 0; i < newSize; i++){
        for (int j = 0; j < newSize; j++){
            A11[i*newSize + j] = A[i*n + j];
            A12[i*newSize + j] = A[i*n + j + newSize];
            A21[i*newSize + j] = A[(i + newSize)*n + j];
            A22[i*newSize + j] = A[(i + newSize)*n + j + newSize];
            
            B11[i*newSize + j] = B[i*n + j];
            B12[i*newSize + j] = B[i*n + j + newSize];
            B21[i*newSize + j] = B[(i + newSize)*n + j];
            B22[i*newSize + j] = B[(i + newSize)*n + j + newSize];
        }
    }
    
    // Allocate temporary matrices for Strassen
    double *M1 = (double*)malloc(size * sizeof(double));
    double *M2 = (double*)malloc(size * sizeof(double));
    double *M3 = (double*)malloc(size * sizeof(double));
    double *M4 = (double*)malloc(size * sizeof(double));
    double *M5 = (double*)malloc(size * sizeof(double));
    double *M6 = (double*)malloc(size * sizeof(double));
    double *M7 = (double*)malloc(size * sizeof(double));
    
    double *T1 = (double*)malloc(size * sizeof(double));
    double *T2 = (double*)malloc(size * sizeof(double));
    double *T3 = (double*)malloc(size * sizeof(double));
    double *T4 = (double*)malloc(size * sizeof(double));
    double *T5 = (double*)malloc(size * sizeof(double));
    double *T6 = (double*)malloc(size * sizeof(double));
    double *T7 = (double*)malloc(size * sizeof(double));
    double *T8 = (double*)malloc(size * sizeof(double));
    double *T9 = (double*)malloc(size * sizeof(double));
    double *T10 = (double*)malloc(size * sizeof(double));
    
    // Build Strassen subexpressions:
    add_matrix(A11, A22, T1, newSize);
    add_matrix(B11, B22, T2, newSize);
    strassen_serial(T1, T2, M1, newSize);
    
    add_matrix(A21, A22, T3, newSize);
    strassen_serial(T3, B11, M2, newSize);
    
    sub_matrix(B12, B22, T4, newSize);
    strassen_serial(A11, T4, M3, newSize);
    
    sub_matrix(B21, B11, T5, newSize);
    strassen_serial(A22, T5, M4, newSize);
    
    add_matrix(A11, A12, T6, newSize);
    strassen_serial(T6, B22, M5, newSize);
    
    sub_matrix(A21, A11, T7, newSize);
    add_matrix(B11, B12, T8, newSize);
    strassen_serial(T7, T8, M6, newSize);
    
    sub_matrix(A12, A22, T9, newSize);
    add_matrix(B21, B22, T10, newSize);
    strassen_serial(T9, T10, M7, newSize);
    
    // Combine sub-results into C
    double *C11 = (double*)malloc(size * sizeof(double));
    double *C12 = (double*)malloc(size * sizeof(double));
    double *C21 = (double*)malloc(size * sizeof(double));
    double *C22 = (double*)malloc(size * sizeof(double));
    
    add_matrix(M1, M4, C11, newSize);
    sub_matrix(C11, M5, C11, newSize);
    add_matrix(C11, M7, C11, newSize);
    
    add_matrix(M3, M5, C12, newSize);
    
    add_matrix(M2, M4, C21, newSize);
    
    sub_matrix(M1, M2, C22, newSize);
    add_matrix(C22, M3, C22, newSize);
    add_matrix(C22, M6, C22, newSize);
    
    // Copy submatrices back to C
    for (int i = 0; i < newSize; i++){
        for (int j = 0; j < newSize; j++){
            C[i*n + j] = C11[i*newSize + j];
            C[i*n + j + newSize] = C12[i*newSize + j];
            C[(i+newSize)*n + j] = C21[i*newSize + j];
            C[(i+newSize)*n + j + newSize] = C22[i*newSize + j];
        }
    }
    
    // Free allocated memory
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(T1); free(T2); free(T3); free(T4); free(T5); free(T6);
    free(T7); free(T8); free(T9); free(T10);
    free(C11); free(C12); free(C21); free(C22);
}

void add_matrix_stride(double *A, int strideA, double *B, int strideB, double *C, int strideC, int n) {
    /*This is function for improved verison of serila strassen functions*/
    
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            C[i*strideC + j] = A[i*strideA + j] + B[i*strideB + j];
        }
    }
}

void sub_matrix_stride(double*A, int strideA, double* B, int strideB, double* C, int strideC, int n){
    for(int i =0; i<n;i++){
        for(int j=0; j<n;j++){
            C[i*strideC + j] = A[i*strideA + j] - B[i*strideB + j];
        }
    }
}

void multiply_standard_stride(double const* restrict A, int strideA,
                            double const* restrict B, int strideB,
                            double* restrict C, int strideC,
                              int n) {
    for(int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * strideC + j] = 0;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a = A[i * strideA + k];
            for (int j = 0; j < n; j++) {
                C[i * strideC + j] += a * B[k * strideB + j];
            }
        }
    }
}


static void strassen_helper_func(double*A, int strideA,
                                double*B, int strideB,
                                double*C, int strideC,
                                int n, double** mem_ptr){
    if(n<=BASE_SIZE){
        multiply_standard_stride(A, strideA, B, strideB, C, strideC, n);
        return;
    }

    const int new_n = n/2;;
    const size_t sub_size = new_n * new_n;

    // Store the currnt pointer for the next usage
    // Remember that, when applying the recursion, the computer uses the DFS,
    // Hence, if the pointer on the one leaf is used, and the leaf finishes its work,
    // the pointer will be used for the next leaf.
    // It is safe to use the pointer, because the pointer is not used in the current leaf.
    double* const curr_ptr = *mem_ptr;

    // Allocate temporary matrices
    double *T1 = *mem_ptr; *mem_ptr += sub_size;
    double *T2 = *mem_ptr; *mem_ptr += sub_size;
    double *T3 = *mem_ptr; *mem_ptr += sub_size;
    double *T4 = *mem_ptr; *mem_ptr += sub_size;
    double *T5 = *mem_ptr; *mem_ptr += sub_size;
    double *T6 = *mem_ptr; *mem_ptr += sub_size;
    double *T7 = *mem_ptr; *mem_ptr += sub_size;
    double *M1 = *mem_ptr; *mem_ptr += sub_size;
    double *M2 = *mem_ptr; *mem_ptr += sub_size;
    double *M3 = *mem_ptr; *mem_ptr += sub_size;
    double *M4 = *mem_ptr; *mem_ptr += sub_size;
    double *M5 = *mem_ptr; *mem_ptr += sub_size;
    double *M6 = *mem_ptr; *mem_ptr += sub_size;
    double *M7 = *mem_ptr; *mem_ptr += sub_size;


    // 定义子矩阵视图（无数据拷贝）
    double *A11 = A;
    double *A12 = A + new_n;
    double *A21 = A + new_n * strideA;
    double *A22 = A21 + new_n;
    
    double *B11 = B;
    double *B12 = B + new_n;
    double *B21 = B + new_n * strideB;
    double *B22 = B21 + new_n;


    // M1 = (A11 + A22)(B11 + B22)
    add_matrix_stride(A11, strideA, A22, strideA, T1, new_n, new_n);
    add_matrix_stride(B11, strideB, B22, strideB, T2, new_n, new_n);
    strassen_helper_func(T1, new_n, T2, new_n, M1, new_n, new_n, mem_ptr);

    // M2 = (A21 + A22)B11
    add_matrix_stride(A21, strideA, A22, strideA, T3, new_n, new_n);
    strassen_helper_func(T3, new_n, B11, strideB, M2, new_n, new_n, mem_ptr);

    // M3 = A11(B12 - B22)
    sub_matrix_stride(B12, strideB, B22, strideB, T4, new_n, new_n);
    strassen_helper_func(A11, strideA, T4, new_n, M3, new_n, new_n, mem_ptr);

    // M4 = A22(B21 - B11)
    sub_matrix_stride(B21, strideB, B11, strideB, T5, new_n, new_n);
    strassen_helper_func(A22, strideA, T5, new_n, M4, new_n, new_n, mem_ptr);

    // M5 = (A11 + A12)B22
    add_matrix_stride(A11, strideA, A12, strideA, T6, new_n, new_n);
    strassen_helper_func(T6, new_n, B22, strideB, M5, new_n, new_n, mem_ptr);

    // M6 = (A21 - A11)(B11 + B12)
    sub_matrix_stride(A21, strideA, A11, strideA, T7, new_n, new_n);
    add_matrix_stride(B11, strideB, B12, strideB, T1, new_n, new_n);
    strassen_helper_func(T7, new_n, T1, new_n, M6, new_n, new_n, mem_ptr);

    // M7 = (A12 - A22)(B21 + B22)
    sub_matrix_stride(A12, strideA, A22, strideA, T2, new_n, new_n);
    add_matrix_stride(B21, strideB, B22, strideB, T3, new_n, new_n);
    strassen_helper_func(T2, new_n, T3, new_n, M7, new_n, new_n, mem_ptr);


    // C11 = M1 + M4 - M5 + M7
    add_matrix_stride(M1, new_n, M4, new_n, T1, new_n, new_n);
    sub_matrix_stride(T1, new_n, M5, new_n, T2, new_n, new_n);
    add_matrix_stride(T2, new_n, M7, new_n, T1, new_n, new_n);
    
    // C12 = M3 + M5
    add_matrix_stride(M3, new_n, M5, new_n, T3, new_n, new_n);
    
    // C21 = M2 + M4 
    add_matrix_stride(M2, new_n, M4, new_n, T4, new_n, new_n);
    
    // C22 = M1 - M2 + M3 + M6
    sub_matrix_stride(M1, new_n, M2, new_n, T5, new_n, new_n);
    add_matrix_stride(T5, new_n, M3, new_n, T6, new_n, new_n);
    add_matrix_stride(T6, new_n, M6, new_n, T7, new_n, new_n);

    // 合并结果到C矩阵
    for (int i = 0; i < new_n; i++) {
        memcpy(&C[i*strideC], &T1[i*new_n], new_n*sizeof(double));
        memcpy(&C[i*strideC + new_n], &T3[i*new_n], new_n*sizeof(double));
        memcpy(&C[(i+new_n)*strideC], &T4[i*new_n], new_n*sizeof(double));
        memcpy(&C[(i+new_n)*strideC + new_n], &T7[i*new_n], new_n*sizeof(double));
    }

    *mem_ptr = curr_ptr;
}


void strassen_serial_optimized(double *A, double *B, double *C, int n) {
    size_t total_mem = 28 * n * n * sizeof(double);
    double *mem_pool = (double*)aligned_alloc(64, total_mem); // 64字节对齐
    if (!mem_pool) return;
    
    memset(C, 0, n*n*sizeof(double)); // 清零结果矩阵
    
    double *mem_ptr = mem_pool;
    strassen_helper_func(A, n, B, n, C, n, n, &mem_ptr);
    
    free(mem_pool);
}