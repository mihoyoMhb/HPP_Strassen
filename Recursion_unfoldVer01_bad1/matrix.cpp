#include "matrix.h"
#include <immintrin.h>   // for AVX instructions if needed
#include <iostream>
#include <cstring>       // For memset
#include <algorithm>
#include <cmath>
#include <stack>
#include <mutex>
#include <omp.h>

// Global task stack for parallel execution
std::stack<TaskFrame> TaskStack;
// Global mutex for thread safety
std::mutex task_mutex;


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

size_t calculate_total_mem(int n) {
    size_t total = 0;
    for (int i = 1; i <= n; i++) {
        total += pow(7, i-1) * 21 * n * n /(pow(2, 2*i));
    }
    return total;
}

size_t get_layer_offset(int current_n, int initial_n) {
    size_t offset = 0;
    
    // // Calculate memory offset for the current recursion layer
    // while (n > current_n) {
    //     offset += pow(7, i) * 21 * n * n /(pow(2, 2*i));
    //     n /= 2;
    // }
    // return offset;
    int t = log2(initial_n / current_n);
    for(int i = 1; i <= t; i++) {
        offset += pow(7, i-1) * 21 * (current_n/2) * (current_n/2) /(pow(2, 2*i));
    }
    return offset;
}

void compute_M_matrix(int m_idx, TaskFrame* frame, double* temp) {
    int n = frame->n;
    int new_n = n / 2;
    
    // Set up pointers for the submatrices of A and B
    double *A = frame->A;
    double *B = frame->B;
    int strideA = frame->strideA;
    int strideB = frame->strideB;
    
    // Pointers to quadrants in A and B
    double *A11 = A;
    double *A12 = A + new_n;
    double *A21 = A + new_n * strideA;
    double *A22 = A21 + new_n;
    
    double *B11 = B;
    double *B12 = B + new_n;
    double *B21 = B + new_n * strideB;
    double *B22 = B21 + new_n;
    
    // Calculate M values according to Strassen algorithm
    TaskFrame newFrame;
    newFrame.n = new_n;
    newFrame.strideC = new_n;
    newFrame.stage = 0;
    
    // 根据m_idx计算独立的临时空间偏移
    const size_t sub_size = new_n * new_n;
    const int temp_offset = 7 * sub_size + 2 * (m_idx-1) * sub_size;
    double* S1 = temp + temp_offset;
    double* S2 = S1 + sub_size;
    
    switch(m_idx) {
        case 1: // M1 = (A11 + A22) * (B11 + B22)
            add_matrix_stride(A11, strideA, A22, strideA, S1, new_n, new_n);
            add_matrix_stride(B11, strideB, B22, strideB, S2, new_n, new_n);
            newFrame.A = S1;
            newFrame.B = S2;
            newFrame.strideA = new_n;
            newFrame.strideB = new_n;
            newFrame.C = temp; // M1 result
            break;
            
        case 2: // M2 = (A21 + A22) * B11
            add_matrix_stride(A21, strideA, A22, strideA, S1, new_n, new_n);
            newFrame.A = S1;
            newFrame.strideA = new_n;
            newFrame.B = B11;
            newFrame.strideB = strideB;
            newFrame.C = temp + sub_size; // M2 result
            break;
            
        case 3: // M3 = A11 * (B12 - B22)
            sub_matrix_stride(B12, strideB, B22, strideB, S1, new_n, new_n);
            newFrame.A = A11;
            newFrame.strideA = strideA;
            newFrame.B = S1;
            newFrame.strideB = new_n;
            newFrame.C = temp + 2 * sub_size; // M3 result
            break;
            
        case 4: // M4 = A22 * (B21 - B11)
            sub_matrix_stride(B21, strideB, B11, strideB, S1, new_n, new_n);
            newFrame.A = A22;
            newFrame.strideA = strideA;
            newFrame.B = S1;
            newFrame.strideB = new_n;
            newFrame.C = temp + 3 * sub_size; // M4 result
            break;
            
        case 5: // M5 = (A11 + A12) * B22
            add_matrix_stride(A11, strideA, A12, strideA, S1, new_n, new_n);
            newFrame.A = S1;
            newFrame.strideA = new_n;
            newFrame.B = B22;
            newFrame.strideB = strideB;
            newFrame.C = temp + 4 * sub_size; // M5 result
            break;
            
        case 6: // M6 = (A21 - A11) * (B11 + B12)
            sub_matrix_stride(A21, strideA, A11, strideA, S1, new_n, new_n);
            add_matrix_stride(B11, strideB, B12, strideB, S2, new_n, new_n);
            newFrame.A = S1;
            newFrame.strideA = new_n;
            newFrame.B = S2;
            newFrame.strideB = new_n;
            newFrame.C = temp + 5 * sub_size; // M6 result
            break;
            
        case 7: // M7 = (A12 - A22) * (B21 + B22)
            sub_matrix_stride(A12, strideA, A22, strideA, S1, new_n, new_n);
            add_matrix_stride(B21, strideB, B22, strideB, S2, new_n, new_n);
            newFrame.A = S1;
            newFrame.strideA = new_n;
            newFrame.B = S2;
            newFrame.strideB = new_n;
            newFrame.C = temp + 6 * sub_size; // M7 result
            break;
    }
    
    multiply_standard_stride_p(newFrame.A, newFrame.strideA,
                            newFrame.B, newFrame.strideB,
                            newFrame.C, newFrame.strideC,
                            new_n);

}

void combineResults(TaskFrame* frame, double* temp) {
    int n = frame->n;
    int new_n = n / 2;
    double *C = frame->C;
    int strideC = frame->strideC;
    
    // Pointers to quadrants in C
    double *C11 = C;
    double *C12 = C + new_n;
    double *C21 = C + new_n * strideC;
    double *C22 = C + new_n * strideC + new_n;
    
    size_t sub_size = new_n * new_n;
    double *M1 = temp;
    double *M2 = temp + sub_size;
    double *M3 = temp + 2 * sub_size;
    double *M4 = temp + 3 * sub_size;
    double *M5 = temp + 4 * sub_size;
    double *M6 = temp + 5 * sub_size;
    double *M7 = temp + 6 * sub_size;
    
    // C11 = M1 + M4 - M5 + M7
    for (int i = 0; i < new_n; i++) {
        for (int j = 0; j < new_n; j++) {
            C11[i * strideC + j] = M1[i * new_n + j] + M4[i * new_n + j] - 
                                   M5[i * new_n + j] + M7[i * new_n + j];
        }
    }
    
    // C12 = M3 + M5
    for (int i = 0; i < new_n; i++) {
        for (int j = 0; j < new_n; j++) {
            C12[i * strideC + j] = M3[i * new_n + j] + M5[i * new_n + j];
        }
    }
    
    // C21 = M2 + M4
    for (int i = 0; i < new_n; i++) {
        for (int j = 0; j < new_n; j++) {
            C21[i * strideC + j] = M2[i * new_n + j] + M4[i * new_n + j];
        }
    }
    
    // C22 = M1 + M3 - M2 + M6
    for (int i = 0; i < new_n; i++) {
        for (int j = 0; j < new_n; j++) {
            C22[i * strideC + j] = M1[i * new_n + j] + M3[i * new_n + j] - 
                                   M2[i * new_n + j] + M6[i * new_n + j];
        }
    }
}



void createSubtasks(TaskFrame* frame, double* temp) {
    // Create tasks for each M computation
    #pragma omp taskgroup
    {
        #pragma omp task shared(temp)
        compute_M_matrix(1, frame, temp+ 0*21*(frame->n/2)*(frame->n/2));
        
        #pragma omp task shared(temp)
        compute_M_matrix(2, frame, temp+ 1*21*(frame->n/2)*(frame->n/2));
        
        #pragma omp task shared(temp)
        compute_M_matrix(3, frame, temp+ 2*21*(frame->n/2)*(frame->n/2));
        
        #pragma omp task shared(temp)
        compute_M_matrix(4, frame, temp+ 3*21*(frame->n/2)*(frame->n/2));
        
        #pragma omp task shared(temp)
        compute_M_matrix(5, frame, temp+ 4*21*(frame->n/2)*(frame->n/2));
        
        #pragma omp task shared(temp)
        compute_M_matrix(6, frame, temp+ 5*21*(frame->n/2)*(frame->n/2));
        
        #pragma omp task shared(temp)
        compute_M_matrix(7, frame, temp+ 6*21*(frame->n/2)*(frame->n/2));
    }
    
    // Update the frame for the combine stage and push back to stack
    frame->stage = 1;
    
    #pragma omp critical
    {
        TaskStack.push(*frame);
    }
}






void process_strassen_layer(TaskFrame* frame, double* mem_pool) {
    // Calculate dimensions for this layer
    int new_n = frame->n / 2;
    
    // 通过frame获取initial_n
    int initial_n = frame->initial_n;
    
    // 计算内存偏移（原get_layer_offset函数可以保留，但需要修改）
    size_t offset = 0;
    int current_n = initial_n;
    while (current_n > frame->n) {
        offset += 21 * (current_n/2) * (current_n/2);
        current_n /= 2;
    }
    
    double* temp_blocks = mem_pool + offset;
    
    // Process according to the current stage
    switch (frame->stage) {
        case 0: // Initial decomposition stage
            createSubtasks(frame, temp_blocks);
            break;
            
        case 1: // Combine results stage
            combineResults(frame, temp_blocks);
            break;
    }
}






void strassen_loop_parallel(double* A, double* B, double* C, int n) {
    // For small matrices, use standard multiplication
    if (n <= BASE_SIZE) {
        multiply_standard_stride_p(A, n, B, n, C, n, n);
        return;
    }
    // Before allocating memory, check if n is too big, if so, use standard multiplication
    // int check = n/BASE_SIZE;
    // if(log2(check) > 5) {
    //     std::cerr << "Matrix size too large, using standard multiplication" << std::endl;
    //     //multiply_standard_stride_p(A, n, B, n, C, n, n);
    //     std::cout << "Done!" << std::endl;
    //     return;
    // }

    int recussion_depth = log2(n / BASE_SIZE);
    if (recussion_depth > 5) {
        std::cerr << "Matrix size too large, using standard multiplication" << std::endl;
        multiply_standard_stride_p(A, n, B, n, C, n, n);
        return;
    }
    // Calculate required memory size
    size_t total_mem = calculate_total_mem(recussion_depth);
    
    // Allocate memory pool with proper alignment for SIMD operations
    double *mem_pool = static_cast<double*>(aligned_alloc(64, total_mem * sizeof(double)));
    if (!mem_pool) {
        std::cerr << "Memory allocation failed, using standard multiplication" << std::endl;
        multiply_standard_stride_p(A, n, B, n, C, n, n);
        return;
    }
    
    // Initialize the memory pool
    memset(mem_pool, 0, total_mem * sizeof(double));
    std::cout << "Memory pool allocated at: " << mem_pool << std::endl;
    std::cout << "Total memory size: " << total_mem * sizeof(double) << " bytes" << std::endl; 
//     // Reset the task stack
//     while (!TaskStack.empty()) {
//         TaskStack.pop();
//     }
    
//     #pragma omp parallel
//     #pragma omp single
//     {
//         // Initialize with the root task
//         TaskFrame root;
//         root.A = A;
//         root.B = B;
//         root.C = C;
//         root.strideA = n;
//         root.strideB = n;
//         root.strideC = n;
//         root.n = n;
//         root.stage = 0;
//         root.initial_n = n;

//         #pragma omp critical
//         {
//             TaskStack.push(root);
//         }
        
//         // Process tasks until the stack is empty
//         while(!TaskStack.empty()) {
//             TaskFrame current_frame;
            
//             #pragma omp critical
//             {
//                 if (!TaskStack.empty()) {
//                     current_frame = TaskStack.top();
//                     TaskStack.pop();
//                 }
//             }
            
//                 // For larger matrices, apply the Strassen algorithm
//                 process_strassen_layer(&current_frame, mem_pool);

//         }
//     }
    
//     // Free the memory pool
//     free(mem_pool);
}

void strassen_parallel_optimized(double *A, double *B, double *C, int n) {
    int recussion_depth = log2(n / BASE_SIZE);
    // if (recussion_depth > 5) {
    //     std::cerr << "Matrix size too large, using standard multiplication" << std::endl;
    //     multiply_standard_stride_p(A, n, B, n, C, n, n);
    //     return;
    // }
    size_t total_mem = calculate_total_mem(recussion_depth);
    double *mem_pool = static_cast<double*>(aligned_alloc(64, total_mem * sizeof(double)));
    if (!mem_pool) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return;
    }
    memset(mem_pool, 0, total_mem * sizeof(double));
    std::cout << "Memory pool allocated at: " << mem_pool << std::endl;
    std::cout << "Total memory size: " << total_mem * sizeof(double) << " bytes" << std::endl; 
}