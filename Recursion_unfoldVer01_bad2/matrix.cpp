#include "matrix.h"
#include <immintrin.h>   // for AVX instructions if needed
#include <iostream>
#include <cstring>       // For memset
#include <algorithm>
#include <cmath>
#include <stack>
#include <mutex>
#include <omp.h>
#include <memory>
// Global task stack for parallel execution
std::stack<TaskFrame> TaskStack;
// Global mutex for thread safety
std::mutex task_mutex;

// Add two matrices: C = A + B
void add_matrix(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            C[i*n + j] = A[i*n + j] + B[i*n + j];
        }
    }
}

// Subtract two matrices: C = A - B
void sub_matrix(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            C[i*n + j] = A[i*n + j] - B[i*n + j];
        }
    }
}

void multiply_standard_stride_p(const double* A, int strideA,
                             const double* B, int strideB,
                             double* C, int strideC, int n) {
    // Initialize C to zero
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * strideC + j] = 0.0;
        }
    }
    
    // Standard matrix multiplication with strides
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[i * strideA + k];
            for (int j = 0; j < n; j++) {
                C[i * strideC + j] += a_ik * B[k * strideB + j];
            }
        }
    }
}

void add_matrix_stride(const double* A, int strideA,
                      const double* B, int strideB,
                      double* C, int strideC, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * strideC + j] = A[i * strideA + j] + B[i * strideB + j];
        }
    }
}

void sub_matrix_stride(const double* A, int strideA,
                      const double* B, int strideB,
                      double* C, int strideC, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * strideC + j] = A[i * strideA + j] - B[i * strideB + j];
        }
    }
}


size_t calculate_total_mem(int recursion_depth, int initial_n) {
    size_t total = 0;
    // i 表示当前递归层级（1 表示第一层，依此类推）
    for (int i = 1; i <= recursion_depth; i++) {
        int layer_size = initial_n / (1 << i);  // 子矩阵尺寸逐层减小
        size_t num_tasks = static_cast<size_t>(pow(7, i - 1));
        size_t layer_mem = num_tasks * 29 * layer_size * layer_size;
        total += layer_mem;
    }
    return total;
}

size_t get_layer_offset(int current_n, int initial_n) {
    size_t offset = 0;
    // t 为当前递归层数：例如 current_n = 512/2^t
    int t = static_cast<int>(log2(initial_n / current_n));
    // 累加前 t-1 层的内存总量
    for (int i = 1; i <= t; i++) {
        int layer_size = initial_n / (1 << i);  // 每层尺寸正确缩小
        size_t num_tasks = static_cast<size_t>(pow(7, i - 1));
        offset += num_tasks * 29 * layer_size * layer_size;
    }
    return offset;
}


void strassen_parallel_optimized(double *A, double *B, double *C, int n) {

    int recursion_depth = log2(n / BASE_SIZE);


    if (recursion_depth > 5) {
        std::cerr << "Matrix size too large, using standard multiplication" << std::endl;
        multiply_standard_stride_p(A, n, B, n, C, n, n);
        return;
    }


    size_t total_mem = calculate_total_mem(recursion_depth, n);
    std::unique_ptr<double, decltype(&std::free)> mem_pool_ptr(
        static_cast<double*>(std::aligned_alloc(64, total_mem * sizeof(double))),
        std::free
    );
    if (!mem_pool_ptr) {
        throw std::bad_alloc();
    }
    std::memset(C, 0, n * n * sizeof(double));
    std::cout << "Memory pool allocated at: " << mem_pool_ptr.get() << std::endl;
    std::cout << "Total memory size: " << total_mem * sizeof(double) << " bytes" << std::endl; 

    // Start parallel region
    double* mem_ptr = mem_pool_ptr.get();
    strassen_helper_func(A, B, C, n, n, 0, &mem_ptr);
}

void strassen_helper_func(const double * A,
    const double * B,
    double * C,
    int n,
    const int initial_n,
    int midx, 
    double ** mem_ptr)
{
    
    if (n <= BASE_SIZE) {
        multiply_standard_stride_p(A, n, B, n, C, n, n);
        return;
    }
    const int newSize = n / 2;
    const size_t sub_size = newSize * newSize;
    
    // Get the offset for the current recursion layer
    size_t offset = get_layer_offset(n, initial_n);
    size_t local_block = 29 * sub_size;
    offset += midx * local_block;
    // Find location of M1-M7, T1-T7 and C11-C22
    // Attention: check!!!
    double *M1 = *mem_ptr + offset;
    // std::cout << "M1: " << M1 << std::endl;
    double *M2 = M1 + sub_size;
    // std::cout << "M2: " << M2 << std::endl;
    // std::cout << "M2 - M1: " << M2 - M1 << std::endl;
    double *M3 = M2 + sub_size;
    double *M4 = M3 + sub_size;
    double *M5 = M4 + sub_size;
    double *M6 = M5 + sub_size;
    double *M7 = M6 + sub_size;
    double *T1 = M7 + sub_size;
    double *T2 = T1 + sub_size;
    double *T3 = T2 + sub_size;
    double *T4 = T3 + sub_size;
    double *T5 = T4 + sub_size;
    double *T6 = T5 + sub_size;
    double *T7 = T6 + sub_size;
    double *T8 = T7 + sub_size;
    double *T9 = T8 + sub_size;
    double *T10 = T9 + sub_size;
    double *C11 = T10 + sub_size;
    double *C12 = C11 + sub_size;
    double *C21 = C12 + sub_size;
    double *C22 = C21 + sub_size;
    // Find location of A11-A22, B11-B22
    double * A11 =  C22 + sub_size;
    double * A12 = A11 + sub_size;
    double * A21 = A12 + sub_size;
    double * A22 = A21 + sub_size;
    double * B11 = A22 + sub_size;
    double * B12 = B11 + sub_size;
    double * B21 = B12 + sub_size;
    double * B22 = B21 + sub_size;
    // std::cout << "B22: " << B22 << std::endl;
    // std::cout << "B22 - M1: " << B22 - M1 << std::endl;
    // std::cout << "All pointers initialized!" << std::endl;
    
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
    // std::cout << "After second check!" << std::endl;
    // Build Strassen subexpressions:
    add_matrix(A11, A22, T1, newSize);
    add_matrix(B11, B22, T2, newSize);
    add_matrix(A21, A22, T3, newSize);
    sub_matrix(B12, B22, T4, newSize);
    sub_matrix(B21, B11, T5, newSize);
    add_matrix(A11, A12, T6, newSize);
    sub_matrix(A21, A11, T7, newSize);
    add_matrix(B11, B12, T8, newSize);
    sub_matrix(A12, A22, T9, newSize);
    add_matrix(B21, B22, T10, newSize);

    // Start parallel region
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Create tasks for each multiplication
            #pragma omp task shared(M1) firstprivate(T1, T2, newSize)
            {
                strassen_helper_func(T1, T2, M1, newSize, initial_n, 0, mem_ptr);
            }
            #pragma omp task shared(M2) firstprivate(T3, B11, newSize)
            {
                strassen_helper_func(T3, B11, M2, newSize, initial_n, 1, mem_ptr);
            }
            #pragma omp task shared(M3) firstprivate(A11, T4, newSize)
            {
                strassen_helper_func(A11, T4, M3, newSize, initial_n, 2, mem_ptr);
            }
            #pragma omp task shared(M4) firstprivate(A22, T5, newSize)
            {
                strassen_helper_func(A22, T5, M4, newSize, initial_n, 3, mem_ptr);
            }
            #pragma omp task shared(M5) firstprivate(T6, B22, newSize)
            {
                strassen_helper_func(T6, B22, M5, newSize, initial_n, 4, mem_ptr);
            }
            #pragma omp task shared(M6) firstprivate(T7,T8,newSize)
            {
                strassen_helper_func(T7,T8,M6,newSize ,initial_n ,5 ,mem_ptr);
            }
            #pragma omp task shared(M7) firstprivate(T9,T10,newSize)
            {
                strassen_helper_func(T9,T10,M7,newSize ,initial_n ,6 ,mem_ptr);
            }
        } // End of single region

        // Wait for all tasks to complete
        #pragma omp taskwait
    } // End of parallel region

    // Combine results into the C matrix
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


}