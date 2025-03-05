#include "matrix.h"
#include <math.h>

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

// Compare two matrices element-wise; return 1 if all differences <= TOLERANCE, else 0
int compare_matrices(double *C1, double *C2, int n) {
    for (int i = 0; i < n*n; i++){
        if (fabs(C1[i] - C2[i]) > TOLERANCE)
            return 0;
    }
    return 1;
}
