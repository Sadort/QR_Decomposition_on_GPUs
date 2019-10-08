#include "kernel.h"
#include <iostream>
#include <fstream>
using namespace std;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

extern void blcoked_qr_calculate(float *d_A, int m, int n, int r);
extern void unblocked_qr_calculate(float *d_A, int m, int n);

void verify(int m, int n)
{
    float *X, *Y;
    int i;
    int num = m / n;
    X = (float *)malloc(n*n * sizeof(float));
    Y = (float *)malloc(n*n * sizeof(float));
    char filename [50];
    sprintf(filename, "o_%d_%d_%d.txt", num, n, n);
    ifstream in_matlab(filename);
    sprintf(filename, "cuda_%d_%d_%d.txt", num, n, n);
    ifstream in_cuda(filename);
    
    if (in_matlab == NULL || in_cuda == NULL) {
        printf("Error Reading File\n");
        exit(0);
    }
    
    for (i = 0; i < n*n; i++) {
        in_matlab >> X[i];
        in_cuda >> Y[i];
    }
    
    in_matlab.close();
    in_cuda.close();
    
    //printMatrix(m, n, X, m, "matlab_result");
    //printMatrix(m, n, Y, m, "cuda_result");
    int flag = 0;
    for (i = 0; i < n*n; i++) {
        if ((X[i] + Y[i]) * (X[i] + Y[i]) > 0.0001 && (X[i] - Y[i]) * (X[i] - Y[i]) > 0.0001) {
            printf("row %d col %d, %f, %f\n", i/n, i%n, X[i], Y[i]);
            flag = 1;
        }
    }
    if(flag == 1)
    {
        printf("Verification Failed.\n\n");
        return;
    }
    printf("Verification complete.\n\n");
}

int main()
{
    int m = 512, n = 256, i, j;
    int r = 64;
    int num = m / n;
    char filename [50];
//    float A[m*n] = { 1, -1, 4, 1,
//                     1, 4, -2, 1,
//                     1, 4, 2, -1,
//                     1, -1, 0, 1 };
    float *A, *d_A;
    printf("Matrix size %i x %i.\n\n", m, n);
    
    A = (float *)malloc(m*n * sizeof(float));
    
    sprintf(filename, "i_%d_%d_%d.txt", num, n, n);
    ifstream in(filename);
    if (in == NULL) {
        printf("Error Reading File\n");
        exit(0);
    }
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            in >> A[IDX2C(i,j,m)];
        }
    }
    in.close();

    cudaError_t cudaStat;
    cudaStat = cudaMalloc((void**)&d_A,sizeof(float)*m*n);
    cudaStat = cudaMemcpy(d_A, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0, totalmseconds = 0;
    int iterations = 20;
    for(i = 0; i < iterations; i++)
    {
        cudaEventRecord(start);

//        unblocked_qr_calculate(d_A, m, n);

        blocked_qr_calculate(d_A, m, n, r);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalmseconds += milliseconds;
        if(i == iterations - 1) break;
        cudaMemcpy( d_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice );
    }

    printf("Elapsed time: %2.13f ms.", totalmseconds / iterations);

    cudaStat = cudaMemcpy(A, d_A, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    sprintf(filename, "cuda_%d_%d_%d.txt", num, n, n);
    ofstream out(filename);
    if (out.is_open()) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                out << A[IDX2C(i,j,m)] << " ";
            }
        }
        out.close();
    }
    
    verify(m, n);
    
//    for (i = 0; i < m*n; i++) {
//        printf("%f ", A[i]);
//    }
//    printf("\n");
    
    cudaFree(d_A);

    return 0;
}
