#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "matrix.h"
// Serial Strassen's algorithm (recursive, without OpenMP tasks)

void add_matrix_stride(double const* restrict A, int strideA,
    double const* restrict B, int strideB,
    double* restrict C, int strideC,
      int n) {
    /*This is function for improved verison of serila strassen functions*/
    
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            C[i*strideC + j] = A[i*strideA + j] + B[i*strideB + j];
        }
    }
}

void sub_matrix_stride(double const* restrict A, int strideA,
    double const* restrict B, int strideB,
    double* restrict C, int strideC,
      int n){
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
    // #pragma omp parallel for 
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