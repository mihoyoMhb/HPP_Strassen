#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <math.h>

// Macro definitions
#define BASE_SIZE 64      // When n <= BASE_SIZE, use standard multiplication
#define TOLERANCE 1e-6    // Tolerance for comparing floating-point results

// --- Common Matrix Operations ---
void add_matrix(double *A, double *B, double *C, int n);
void sub_matrix(double *A, double *B, double *C, int n);
int compare_matrices(double *C1, double *C2, int n);

// --- Serial Versions ---
void multiply_standard_serial(double *A, double *B, double *C, int n);
void strassen_serial(double *A, double *B, double *C, int n);

// --- Parallel Versions ---
void multiply_standard_parallel(double *A, double *B, double *C, int n);
void strassen_parallel(double *A, double *B, double *C, int n);

#endif // MATRIX_H
