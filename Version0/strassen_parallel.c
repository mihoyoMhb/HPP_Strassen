#include "matrix.h"
#include <stdlib.h>
#include <omp.h>


void strassen_parallel(const double *restrict A,
    const double *restrict B,
    double *restrict C,
    int n) {
    if (n <= BASE_SIZE) {
        // For small matrices, fall back to serial standard multiplication
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
    
    // Allocate temporary matrices and results for Strassen's algorithm
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
    
    // 优化点：并行计算所有临时子表达式 T1-T10
    #pragma omp taskgroup
    {
        #pragma omp task shared(T1) firstprivate(A11, A22, newSize)
        {
            add_matrix(A11, A22, T1, newSize);
        }
        
        #pragma omp task shared(T2) firstprivate(B11, B22, newSize)
        {
            add_matrix(B11, B22, T2, newSize);
        }
        
        #pragma omp task shared(T3) firstprivate(A21, A22, newSize)
        {
            add_matrix(A21, A22, T3, newSize);
        }
        
        #pragma omp task shared(T4) firstprivate(B12, B22, newSize)
        {
            sub_matrix(B12, B22, T4, newSize);
        }
        
        #pragma omp task shared(T5) firstprivate(B21, B11, newSize)
        {
            sub_matrix(B21, B11, T5, newSize);
        }
        
        #pragma omp task shared(T6) firstprivate(A11, A12, newSize)
        {
            add_matrix(A11, A12, T6, newSize);
        }
        
        #pragma omp task shared(T7) firstprivate(A21, A11, newSize)
        {
            sub_matrix(A21, A11, T7, newSize);
        }
        
        #pragma omp task shared(T8) firstprivate(B11, B12, newSize)
        {
            add_matrix(B11, B12, T8, newSize);
        }
        
        #pragma omp task shared(T9) firstprivate(A12, A22, newSize)
        {
            sub_matrix(A12, A22, T9, newSize);
        }
        
        #pragma omp task shared(T10) firstprivate(B21, B22, newSize)
        {
            add_matrix(B21, B22, T10, newSize);
        }
    }
    // taskgroup结束后，所有T1-T10的计算已完成
    
    // 现在并行计算7个Strassen乘法
    #pragma omp taskgroup
    {
        #pragma omp task shared(M1) firstprivate(T1, T2, newSize)
        {
            strassen_parallel(T1, T2, M1, newSize);
        }
        
        #pragma omp task shared(M2) firstprivate(T3, B11, newSize)
        {
            strassen_parallel(T3, B11, M2, newSize);
        }
        
        #pragma omp task shared(M3) firstprivate(A11, T4, newSize)
        {
            strassen_parallel(A11, T4, M3, newSize);
        }
        
        #pragma omp task shared(M4) firstprivate(A22, T5, newSize)
        {
            strassen_parallel(A22, T5, M4, newSize);
        }
        
        #pragma omp task shared(M5) firstprivate(T6, B22, newSize)
        {
            strassen_parallel(T6, B22, M5, newSize);
        }
        
        #pragma omp task shared(M6) firstprivate(T7, T8, newSize)
        {
            strassen_parallel(T7, T8, M6, newSize);
        }
        
        #pragma omp task shared(M7) firstprivate(T9, T10, newSize)
        {
            strassen_parallel(T9, T10, M7, newSize);
        }
    }
    // 不需要显式的taskwait，taskgroup保证所有任务完成
    
    // Combine results into matrix C
    double *C11 = (double*)malloc(size * sizeof(double));
    double *C12 = (double*)malloc(size * sizeof(double));
    double *C21 = (double*)malloc(size * sizeof(double));
    double *C22 = (double*)malloc(size * sizeof(double));
    
    // 这里也可以考虑并行，但通常这部分计算量较小
    add_matrix(M1, M4, C11, newSize);
    sub_matrix(C11, M5, C11, newSize);
    add_matrix(C11, M7, C11, newSize);
    
    add_matrix(M3, M5, C12, newSize);
    
    add_matrix(M2, M4, C21, newSize);
    
    sub_matrix(M1, M2, C22, newSize);
    add_matrix(C22, M3, C22, newSize);
    add_matrix(C22, M6, C22, newSize);
    
    // Combine submatrices back into result matrix C
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
