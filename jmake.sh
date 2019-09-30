rm -rf *.out *.o
#/usr/local/cuda-9.0/bin/nvcc -c -I/usr/local/cuda-9.0/include qr_example.cpp
#/usr/local/cuda-9.0/bin/nvcc -o a.out -Xcompiler -fopenmp qr_example.o -L/usr/local/cuda-9.0/lib64 -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcudart -lcublas -lcusolver

#works
#/usr/local/cuda/bin/nvcc -c -I/usr/local/cuda/include qr_example.cpp
#/usr/local/cuda/bin/nvcc -o a.out -Xcompiler -fopenmp qr_example.o -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcudart -lcublas -lcusolver

/usr/local/cuda-9.0/bin/nvcc -c -I/usr/local/cuda/include qr_code.cu
/usr/local/cuda-9.0/bin/nvcc -o a.out -Xcompiler -fopenmp qr_code.o -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcudart -lcublas -lcusolver

rm -rf qr_example.o
