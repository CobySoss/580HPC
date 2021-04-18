/*
 * spmm_csr_driver.cu
 * Copyright (C) 2020
 *  Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#include "mm_helper.hpp"
#include "sparse_representation.hpp"
#include <iostream>

void check_dmat(double* a, double *b, unsigned int n, unsigned int K, bool quit_on_err = true ) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < K; ++k) {
            if(std::abs(a[i * K + k] - b[i * K + k]) > 1e-1) {
                std::cerr << "Possible error at " << i << std::endl;

                if(quit_on_err) {
                    exit(-1);
                }
            }
        }
    }

    if(quit_on_err)
        std::cout << "Verification succeeded\n";
    else
        std::cout << "Check error messages to see if verification succeeded. (No error msg == success)\n";
}

static unsigned int g_seed = 0X4B1D;
inline int fastrand() {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

void init_dmat(double *a, unsigned int n, unsigned int K, double offset) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < K; ++k) {
            a[i * K + k]  = i * K + k + offset;
            //a[i * K + j]  = fastrand() + offset;
        }
    }
}

void print_dmat(double *a, unsigned int n, unsigned int K) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < K; ++j) {
            std::cout << a[i * K + j]   << ' ';
        }
        std::cout << '\n';
    }
}

void print_CSR(CSR &mat) {
    for (unsigned int r = 0; r < mat.nrows; ++r) {
        unsigned int row_start = mat.row_indx[r];
        unsigned int row_end = mat.row_indx[r + 1];
        for (unsigned int j = row_start; j < row_end; ++j) {
            unsigned int col_id = mat.col_id[j];
            double val = mat.values[j];

	    std::cout << r << ' ' << col_id << ' ' <<  val << '\n';
        }
    }
}

void host_csr_spmm(CSR &mat, double * dmat_in, double * dmat_out, unsigned int K) {
    for (unsigned int r = 0; r < mat.nrows; ++r) {
        unsigned int row_start = mat.row_indx[r];
        unsigned int row_end = mat.row_indx[r + 1];

        for (unsigned int k = 0; k < K; ++k) {
            dmat_out[r * K + k] = 0;
        }

        for (unsigned int j = row_start; j < row_end; ++j) {
            unsigned int col_id = mat.col_id[j];
            double val = mat.values[j];

            for (unsigned int k = 0; k < K; ++k) {
                dmat_out[r * K + k] += val * dmat_in[col_id * K + k];
            }
        }

    }
}

__global__ void dev_csr_spmm(CSR mat, double* mat_in, double* mat_out, unsigned int K)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < mat.nrows)
	{
	   unsigned int row_start = mat.row_indx[row];
	   unsigned int row_end = mat.row_indx[row + 1];
	   //printf("%d\n", row_end - row_start);
	   for(unsigned int k = 0; k < K; k++)
	   {
	   	mat_out[row * K + k] = 0;
	   }
	  // printf("%d %d %d\n", row, row_start, row_end);
	   for(unsigned int j = row_start; j < row_end; j++)
	   {
              //printf("are we here!  %d\n", j);  
	      unsigned int col_id = mat.col_id[j];
	      double value = mat.values[j];
	      for (unsigned int k = 0; k < K; ++k) {
                  mat_out[row * K + k] += value * mat_in[col_id * K + k];
              }
	   }
	}
}

int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cerr << "usage ./exec M N" << std::endl;
        exit(-1);
    }

    unsigned int N = std::atoi(argv[2]);
    unsigned int M = std::atoi(argv[1]);
    std::string s;
    int nnz = 0;
    int currRowDensityMod = 0;
    for(int i = 1; i <= M; i ++)
    {
       currRowDensityMod = fastrand() % 10;
       for(int j = 1; j <= N; j++)
       {
	  int modVal = 0;
	  if(currRowDensityMod != 0)
	  {
 	    modVal  = fastrand() % currRowDensityMod;
	  }
       	  if(modVal == 0)
	  {
	     nnz+=1;
	     s += std::to_string(i) + " " + std::to_string(j) + " " + std::to_string(fastrand() % 3) + "\n";
	  }
       }
    }
    std::string header = "\%testfile\n" + std::to_string(M) + " " + std::to_string(N) + " " + std::to_string(nnz) + "\n";
    s = header + s;
    std::cout << s; 
    return 0;
  }
