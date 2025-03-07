#include "matrix.h"
#include <immintrin.h> // for AVX instructions if needed
#include <iostream>
#include <cstring>
// Global task stack for parallel execution
// Global task stack for parallel execution
std::stack<TaskFrame> TaskStack;
// Safer matrix initialization function
void safe_zero_matrix(double* mat, int stride, int n) {
    // Use a single loop to minimize potential issues
    std::fill_n(mat, n * stride, 0.0);
    std::cout << "Matrix zeroed successfully" << std::endl;
}

void multiply_standard_stride_p(const double* A, int strideA,
                             const double* B, int strideB,
                             double* C, int strideC, int n) {
    std::cout << "Standard multiplication with strides, n=" << n << std::endl;
    std::cout << "strideA=" << strideA << ", strideB=" << strideB << ", strideC=" << strideC << std::endl;

    // Safety check - validate parameters
    if (A == nullptr || B == nullptr || C == nullptr) {
        std::cerr << "Error: Null matrix pointer detected!" << std::endl;
        return;
    }

    try {
        // Try a simpler initialization approach
        safe_zero_matrix(C, strideC, n);
        
        // Standard matrix multiplication with strides
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double a_ik = A[i * strideA + k];
                for (int j = 0; j < n; j++) {
                    C[i * strideC + j] += a_ik * B[k * strideB + j];
                }
            }
        }
        
        std::cout << "Multiplication completed" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in multiply_standard_stride_p: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception in multiply_standard_stride_p" << std::endl;
    }
}

// Let's simplify the algorithm for debugging
void strassen_loop_parallel(double* A, double* B, double* C, int n) {
    std::cout << "Starting strassen_loop_parallel with n=" << n << std::endl;
    
    // Input validation
    if (!A || !B || !C) {
        std::cerr << "Error: Invalid matrix pointers!" << std::endl;
        return;
    }
    
    // For debugging, let's temporarily bypass Strassen and use standard multiplication
    std::cout << "Debug mode: Using standard multiplication directly" << std::endl;
    try {
        // Try with smaller chunks to see if we can identify the issue
        const int chunk_size = 32; // Try smaller blocks
        
        // Zero out the result matrix C
        std::fill_n(C, n * n, 0.0);
        
        // Process the matrix in smaller chunks
        for (int bi = 0; bi < n; bi += chunk_size) {
            int block_size_i = std::min(chunk_size, n - bi);
            
            for (int bj = 0; bj < n; bj += chunk_size) {
                int block_size_j = std::min(chunk_size, n - bj);
                
                // Multiply this block
                for (int i = 0; i < block_size_i; i++) {
                    for (int k = 0; k < n; k++) {
                        double a_ik = A[(bi + i) * n + k];
                        for (int j = 0; j < block_size_j; j++) {
                            C[(bi + i) * n + (bj + j)] += a_ik * B[k * n + (bj + j)];
                        }
                    }
                }
                
                std::cout << "Processed block (" << bi << "," << bj << ") successfully" << std::endl;
            }
        }
        
        std::cout << "Standard multiplication completed successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
    }
}
// void multiply_standard_stride_p(const double* A, int strideA,
//                              const double* B, int strideB,
//                              double* C, int strideC, int n) {
//     std::cout << "Standard multiplication with strides, n=" << n << std::endl;
//     std::cout << "strideA=" << strideA << ", strideB=" << strideB << ", strideC=" << strideC << std::endl;

//     // Safety check - validate parameters
//     if (A == nullptr || B == nullptr || C == nullptr) {
//         std::cerr << "Error: Null matrix pointer detected!" << std::endl;
//         return;
//     }

//     // Access the first element to check if pointers are valid
//     volatile double test_a = A[0];
//     volatile double test_b = B[0];
//     volatile double test_c = C[0];  // This will crash if C is invalid
    
//     std::cout << "Matrix pointers are valid" << std::endl;
    
//     // Zero out the C matrix before multiplication
//     for(int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             C[i * strideC + j] = 0.0;
//         }
//     }
    
//     std::cout << "Filled C with zeros" << std::endl;
    
//     // Standard matrix multiplication with strides
//     for (int i = 0; i < n; i++) {
//         for (int k = 0; k < n; k++) {
//             double a_ik = A[i * strideA + k];
//             for (int j = 0; j < n; j++) {
//                 C[i * strideC + j] += a_ik * B[k * strideB + j];
//             }
//         }
//     }
    
//     std::cout << "Multiplication completed" << std::endl;
// }

// void compute_M_matrix(int m_idx, TaskFrame* frame, double* temp) {
//     int n = frame->n;
//     int new_n = n / 2;
    
//     // Set up pointers for the submatrices of A and B
//     double *A = frame->A;
//     double *B = frame->B;
//     int strideA = frame->strideA;
//     int strideB = frame->strideB;
    
//     // Pointers to quadrants in A and B
//     double *A11 = A;
//     double *A12 = A + new_n;
//     double *A21 = A + new_n * strideA;
//     double *A22 = A + new_n * strideA + new_n;
    
//     double *B11 = B;
//     double *B12 = B + new_n;
//     double *B21 = B + new_n * strideB;
//     double *B22 = B + new_n * strideB + new_n;
    
//     // Calculate M values according to Strassen algorithm
//     TaskFrame newFrame;
//     newFrame.n = new_n;
//     newFrame.strideC = new_n;
//     newFrame.stage = 0;
    
//     size_t sub_size = new_n * new_n;
//     double *S1 = temp + 7 * sub_size;  // First temporary space
//     double *S2 = S1 + sub_size;       // Second temporary space
    
//     switch(m_idx) {
//         case 1: // M1 = (A11 + A22) * (B11 + B22)
//             // S1 = A11 + A22
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S1[i * new_n + j] = A11[i * strideA + j] + A22[i * strideA + j];
//                 }
//             }
            
//             // S2 = B11 + B22
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S2[i * new_n + j] = B11[i * strideB + j] + B22[i * strideB + j];
//                 }
//             }
//             newFrame.A = S1;
//             newFrame.B = S2;
//             newFrame.strideA = new_n;
//             newFrame.strideB = new_n;
//             newFrame.C = temp; // M1 result
//             break;
            
//         case 2: // M2 = (A21 + A22) * B11
//             // S1 = A21 + A22
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S1[i * new_n + j] = A21[i * strideA + j] + A22[i * strideA + j];
//                 }
//             }
//             newFrame.A = S1;
//             newFrame.strideA = new_n;
//             newFrame.B = B11;
//             newFrame.strideB = strideB;
//             newFrame.C = temp + sub_size; // M2 result
//             break;
            
//         case 3: // M3 = A11 * (B12 - B22)
//             // S1 = B12 - B22
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S1[i * new_n + j] = B12[i * strideB + j] - B22[i * strideB + j];
//                 }
//             }
//             newFrame.A = A11;
//             newFrame.strideA = strideA;
//             newFrame.B = S1;
//             newFrame.strideB = new_n;
//             newFrame.C = temp + 2 * sub_size; // M3 result
//             break;
            
//         case 4: // M4 = A22 * (B21 - B11)
//             // S1 = B21 - B11
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S1[i * new_n + j] = B21[i * strideB + j] - B11[i * strideB + j];
//                 }
//             }
//             newFrame.A = A22;
//             newFrame.strideA = strideA;
//             newFrame.B = S1;
//             newFrame.strideB = new_n;
//             newFrame.C = temp + 3 * sub_size; // M4 result
//             break;
            
//         case 5: // M5 = (A11 + A12) * B22
//             // S1 = A11 + A12
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S1[i * new_n + j] = A11[i * strideA + j] + A12[i * strideA + j];
//                 }
//             }
//             newFrame.A = S1;
//             newFrame.strideA = new_n;
//             newFrame.B = B22;
//             newFrame.strideB = strideB;
//             newFrame.C = temp + 4 * sub_size; // M5 result
//             break;
            
//         case 6: // M6 = (A21 - A11) * (B11 + B12)
//             // S1 = A21 - A11
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S1[i * new_n + j] = A21[i * strideA + j] - A11[i * strideA + j];
//                 }
//             }
            
//             // S2 = B11 + B12
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S2[i * new_n + j] = B11[i * strideB + j] + B12[i * strideB + j];
//                 }
//             }
//             newFrame.A = S1;
//             newFrame.strideA = new_n;
//             newFrame.B = S2;
//             newFrame.strideB = new_n;
//             newFrame.C = temp + 5 * sub_size; // M6 result
//             break;
            
//         case 7: // M7 = (A12 - A22) * (B21 + B22)
//             // S1 = A12 - A22
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S1[i * new_n + j] = A12[i * strideA + j] - A22[i * strideA + j];
//                 }
//             }
            
//             // S2 = B21 + B22
//             for (int i = 0; i < new_n; i++) {
//                 for (int j = 0; j < new_n; j++) {
//                     S2[i * new_n + j] = B21[i * strideB + j] + B22[i * strideB + j];
//                 }
//             }
//             newFrame.A = S1;
//             newFrame.strideA = new_n;
//             newFrame.B = S2;
//             newFrame.strideB = new_n;
//             newFrame.C = temp + 6 * sub_size; // M7 result
//             break;
//     }
    
//     // Recursively compute M matrix
//     if (new_n <= BASE_SIZE) {
//         multiply_standard_stride_p(newFrame.A, newFrame.strideA, 
//                                 newFrame.B, newFrame.strideB, 
//                                 newFrame.C, newFrame.strideC, 
//                                 new_n);
//     } else {
//         // Recursively use the parallel Strassen algorithm
//         strassen_loop_parallel(newFrame.A, newFrame.B, newFrame.C, new_n);
//     }
// }

// void createSubtasks(TaskFrame* frame, double* temp) {
//     int n = frame->n;
//     int new_n = n / 2;
//     size_t sub_size = new_n * new_n;
    
//     // Create tasks for each M computation
//     #pragma omp task shared(temp)
//     {
//         compute_M_matrix(1, frame, temp);
//     }
    
//     #pragma omp task shared(temp)
//     {
//         compute_M_matrix(2, frame, temp);
//     }
    
//     #pragma omp task shared(temp)
//     {
//         compute_M_matrix(3, frame, temp);
//     }
    
//     #pragma omp task shared(temp)
//     {
//         compute_M_matrix(4, frame, temp);
//     }
    
//     #pragma omp task shared(temp)
//     {
//         compute_M_matrix(5, frame, temp);
//     }
    
//     #pragma omp task shared(temp)
//     {
//         compute_M_matrix(6, frame, temp);
//     }
    
//     #pragma omp task shared(temp)
//     {
//         compute_M_matrix(7, frame, temp);
//     }
    
//     // Create a task to combine results, with dependencies on all M tasks
//     #pragma omp taskwait
    
//     // Update the frame for the combine stage and push back to stack
//     frame->stage = 1;
    
//     #pragma omp critical
//     {
//         TaskStack.push(*frame);
//     }
// }

// void combineResults(TaskFrame* frame, double* temp) {
//     int n = frame->n;
//     int new_n = n / 2;
//     double *C = frame->C;
//     int strideC = frame->strideC;
    
//     // Pointers to quadrants in C
//     double *C11 = C;
//     double *C12 = C + new_n;
//     double *C21 = C + new_n * strideC;
//     double *C22 = C + new_n * strideC + new_n;
    
//     size_t sub_size = new_n * new_n;
//     double *M1 = temp;
//     double *M2 = temp + sub_size;
//     double *M3 = temp + 2 * sub_size;
//     double *M4 = temp + 3 * sub_size;
//     double *M5 = temp + 4 * sub_size;
//     double *M6 = temp + 5 * sub_size;
//     double *M7 = temp + 6 * sub_size;
    
//     // C11 = M1 + M4 - M5 + M7
//     for (int i = 0; i < new_n; i++) {
//         for (int j = 0; j < new_n; j++) {
//             C11[i * strideC + j] = M1[i * new_n + j] + M4[i * new_n + j] - M5[i * new_n + j] + M7[i * new_n + j];
//         }
//     }
    
//     // C12 = M3 + M5
//     for (int i = 0; i < new_n; i++) {
//         for (int j = 0; j < new_n; j++) {
//             C12[i * strideC + j] = M3[i * new_n + j] + M5[i * new_n + j];
//         }
//     }
    
//     // C21 = M2 + M4
//     for (int i = 0; i < new_n; i++) {
//         for (int j = 0; j < new_n; j++) {
//             C21[i * strideC + j] = M2[i * new_n + j] + M4[i * new_n + j];
//         }
//     }
    
//     // C22 = M1 + M3 - M2 + M6
//     for (int i = 0; i < new_n; i++) {
//         for (int j = 0; j < new_n; j++) {
//             C22[i * strideC + j] = M1[i * new_n + j] + M3[i * new_n + j] - M2[i * new_n + j] + M6[i * new_n + j];
//         }
//     }
// }

// size_t get_layer_offset(int current_n, int initial_n) {
//     size_t offset = 0;
//     int n = initial_n;
    
//     // Calculate memory offset for the current recursion layer
//     while (n > current_n) {
//         // Each layer needs 14 temporary matrices of size (n/2)^2
//         offset += 14 * (n / 2) * (n / 2);
//         n /= 2;
//     }
//     return offset;
// }

// void process_strassen_layer(TaskFrame* frame, double* mem_pool) {
//     // Calculate dimensions for this layer
//     int new_n = frame->n / 2;
//     size_t sub_size = new_n * new_n;
    
//     // Allocate memory for the current layer's temporary matrices
//     // We need memory for 7 M matrices and additional scratch space
//     static int initial_n = 0;
//     if (initial_n == 0) {
//         // First call, initialize initial_n
//         initial_n = frame->n;
//     }
    
//     size_t offset = get_layer_offset(frame->n, initial_n);
//     double* temp_blocks = mem_pool + offset;
    
//     std::cout << "Processing Strassen layer with n = " << frame->n << ", offset = " << offset << std::endl;
//     // Process according to the current stage
//     switch (frame->stage) {
//         case 0: // Initial decomposition stage
//             createSubtasks(frame, temp_blocks);
//             break;
            
//         case 1: // Combine results stage
//             combineResults(frame, temp_blocks);
//             break;
//     }
// }

// void strassen_loop_parallel(double* A, double* B, double* C, int n) {
//     // Validate input parameters
//     if (!A || !B || !C || n <= 0) {
//         std::cerr << "Invalid parameters in strassen_loop_parallel: A=" << A 
//                   << ", B=" << B << ", C=" << C << ", n=" << n << std::endl;
//         return;
//     }

//     // Ensure n is a power of 2
//     if ((n & (n-1)) != 0) {
//         std::cerr << "Error: Matrix dimension n=" << n << " is not a power of 2" << std::endl;
//         return;
//     }
    
//     // Allocate memory pool with proper alignment for SIMD operations
//     size_t total_mem = 28 * n * n * sizeof(double);
//     std::cout << "Allocating memory pool of size " << total_mem << " bytes" << std::endl;
    
//     double *mem_pool = (double*)aligned_alloc(64, total_mem);
//     if (!mem_pool) {
//         std::cerr << "Memory allocation failed!" << std::endl;
//         return;
//     }
//     std::cout << "Memory pool allocated at: " << mem_pool << std::endl;
    
//     // Zero out the memory pool to avoid uninitialized values
//     memset(mem_pool, 0, total_mem);
    
//     #pragma omp parallel
//     #pragma omp single
//     {
//         // Initialize with the root task
//         TaskFrame root = {A, B, C, n, n, n, n, 0};
//         std::cout << "Created root task with n=" << n << std::endl;
        
//         #pragma omp critical
//         {
//             TaskStack.push(root);
//         }
        
//         // Process tasks until the stack is empty
//         while(!TaskStack.empty()) {
//             #pragma omp task untied
//             {
//                 TaskFrame current_frame;
                
//                 #pragma omp critical
//                 {
//                     current_frame = TaskStack.top();
//                     TaskStack.pop();
//                 }
                
//                 std::cout << "Processing task with n=" << current_frame.n 
//                           << ", A=" << current_frame.A
//                           << ", B=" << current_frame.B
//                           << ", C=" << current_frame.C
//                           << ", stage=" << current_frame.stage << std::endl;
                          
//                 if(current_frame.n <= BASE_SIZE) {
//                     // For small matrices, use standard multiplication
//                     multiply_standard_stride_p(current_frame.A, current_frame.strideA,
//                                            current_frame.B, current_frame.strideB,
//                                            current_frame.C, current_frame.strideC,
//                                            current_frame.n);
//                 } else {
//                     // For larger matrices, apply the Strassen algorithm
//                     process_strassen_layer(&current_frame, mem_pool);
//                 }
//             }
//         }
//     }
    
//     // Free the memory pool
//     free(mem_pool);
//     std::cout << "Memory pool freed" << std::endl;
// }