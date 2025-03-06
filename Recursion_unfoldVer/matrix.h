#ifndef MATRIX_H
#define MATRIX_H


#include <iostream>
#include <stack>
#include <omp.h>

const int BASE_SIZE = 128;


typedef struct {
    double *A, *B, *C;
    int strideA, strideB, strideC;
    int n; // Current size of the matrix
    int stage; // Current stage of the recursion
} TaskFrame;

struct MatrixBlock {
    double *data;
    int row_stride;
    int col_stride;
};


std::stack<TaskFrame> TaskStack;

void multiply_standard_stride(const double* A, int strideA, 
    const double* B, int strideB,
    double* C, int strideC, int n);

void blockMultiply(MatrixBlock a, MatrixBlock b, MatrixBlock c);


void process_strassen_layer(TaskFrame *frame, double *mem_pool);

size_t get_layer_offset(int current_n, int initial_n);


#endif // MATRIX_H