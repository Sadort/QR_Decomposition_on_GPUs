#include <cmath>
#include <assert.h>
#include "kernel.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

extern void blcoked_qr_calculate(double *d_A, int m, int n, int r);
extern void unblocked_qr_calculate(double *d_A, int m, int n);

void verify(int m, int n)
{
    double *X, *Y;
    int i;
    int num = m / n;
    X = (double *)malloc(n*n * sizeof(double));
    Y = (double *)malloc(n*n * sizeof(double));
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
    double sum = 0;
    double maxerror = 0;
    for (i = 0; i < n*n; i++) {
        double error = std::abs(std::abs(X[i]) - std::abs(Y[i]));
        sum += error;
        if(error > maxerror) maxerror = error;
        if ((X[i] + Y[i]) * (X[i] + Y[i]) > 0.0001 && (X[i] - Y[i]) * (X[i] - Y[i]) > 0.0001) {
            //printf("row %d col %d, %f, %f\n", i/n, i%n, X[i], Y[i]);
            flag += 1;
        }
    }
    printf("sum error: %lf\n", sum);
    printf("max error: %lf\n", maxerror);
    if(flag > 0)
    {
        printf(" Incorrectness = %f, Verification Failed.\n\n", (float)flag/(n*n));
        printf(" flag = %d, Verification Failed.\n\n", flag);
        return;
    }
    printf("Verification complete.\n\n");
}

int main()
{
    int m = 8192, n = 4096, i, j;
//    int r = 4096;
    int r;
    cin >> r;
    int num = m / n;
    char filename [50];
//    float A[m*n] = { 1, -1, 4, 1,
//                     1, 4, -2, 1,
//                     1, 4, 2, -1,
//                     1, -1, 0, 1 };
    double *A, *d_A;
    printf("Matrix size %i x %i. r = %d.\n\n", m, n, r);
    
    A = (double *)malloc(m*n * sizeof(double));
    
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
    cudaStat = cudaMalloc((void**)&d_A,sizeof(double)*m*n);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(d_A, A, sizeof(double)*m*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
/*
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0, totalmseconds = 0;
    int iterations = 10;
    for(i = 0; i < iterations; i++)
    {
        cudaEventRecord(start);
*/
//        unblocked_qr_calculate(d_A, m, n);

    blocked_qr_calculate(d_A, m, n, r);
/*
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalmseconds += milliseconds;
        if(i == iterations - 1) break;
        cudaMemcpy( d_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice );
        cudaDeviceSynchronize();
    }

    printf("Elapsed time: %f ms.", totalmseconds / iterations);
*/
    cudaStat = cudaMemcpy(A, d_A, sizeof(double)*m*n, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat);
    cudaDeviceSynchronize();

    sprintf(filename, "cuda_%d_%d_%d.txt", num, n, n);
    ofstream out(filename);
    if (out.is_open()) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                out << A[IDX2C(i,j,m)] << " ";
            }
            out << endl;
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
