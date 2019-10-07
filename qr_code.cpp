#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cublas_v2.h"

using namespace std;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

extern void blcoked_qr_calculate(float *d_A, int m, int n, int r);
extern void unblocked_qr_calculate(float *d_A, int m, int n);

int main()
{
    int m = 4, n = 4, i;
    int r = 2;
    float A[m*n] = { 1, -1, 4, 1,
                     1, 4, -2, 1,
                     1, 4, 2, -1,
                     1, -1, 0, 1 };
    float *d_A;
    
    

    cudaError_t cudaStat;
    cudaStat = cudaMalloc((void**)&d_A,sizeof(float)*m*n);
    cudaStat = cudaMemcpy(d_A, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    //unblocked_qr_calculate(d_A, m, n);

    blocked_qr_calculate(d_A, m, n, r);

    cudaStat = cudaMemcpy(A, d_A, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (i = 0; i < m*n; i++) {
        printf("%f ", A[i]);
    }
    printf("\n");

    return 0;
}
