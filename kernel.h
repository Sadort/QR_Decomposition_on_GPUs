#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
void unblocked_qr_calculate(double *d_A, int m, int n);
void blocked_qr_calculate(double *d_A, int m, int n, int r);
