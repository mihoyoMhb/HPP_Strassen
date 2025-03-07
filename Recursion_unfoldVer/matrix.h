#ifndef MATRIX_H
#define MATRIX_H

#include <stack>

// Base size for which we switch to standard multiplication
#define BASE_SIZE 512

// Task frame for the Strassen algorithm
struct TaskFrame {
    double* A;
    double* B;
    double* C;
    int strideA;
    int strideB;
    int strideC;
    int n;
    int stage;
};

// Global task stack for parallel execution
extern std::stack<TaskFrame> TaskStack;

// Function declarations
void strassen_loop_parallel(double* A, double* B, double* C, int n);
void process_strassen_layer(TaskFrame* frame, double* mem_pool);
void multiply_standard_stride_p(const double* A, int strideA, const double* B, int strideB, 
                               double* C, int strideC, int n);
void createSubtasks(TaskFrame* frame, double* temp);
void combineResults(TaskFrame* frame, double* temp);
void compute_M_matrix(int m_idx, TaskFrame* frame, double* temp);
size_t get_layer_offset(int current_n, int initial_n);

#endif // MATRIX_H