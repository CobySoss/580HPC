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
        std::cerr << "usage ./exec filename  " << std::endl;
        exit(-1);
    }
    int numRow = 0;
    int numCol = 0;
    double* mat1 = getDenseMat("small.mtx", &numRow, &numCol);    
    CSR mat = read_matrix_market_to_CSR("small.mtx");
    double* mat2 = getDenseMat(mat, &numRow, &numCol);
    printf("r %d c %d\n", numRow, numCol);
    for(unsigned int m = 0; m < numRow; m++)
    {
    	for(unsigned int n = 0; n < numCol; n++)
	{
	   if(mat1[m * numCol + n] - mat2[m * numCol + n]  > .001)
	   {
	   	printf("Not equal\n");
	   }
	}
    }
    int numRowB = 0;
    int numColB = 0;
    CSR matb = read_matrix_market_to_CSR("smallb.mtx");
    double* denseMatB = getDenseMat(matb, &numRowB, &numColB);
    printf("rowb %d colb %d \n", numRowB, numColB);  
    int numRowC = 0;
    int numColC = 0;
    double * cMat = DenseDenseMult(mat1, numRow, numCol, denseMatB, numRowB, numColB, &numRowC,&numColC);
    printf("rowc %d colc %d\n", numRowC, numColC); 
    for(int rc = 0; rc < numRowC; rc++)
    {
    	for(int cc = 0; cc < numColC; cc++)
	{
	    double val = cMat[rc * numColC +cc];
            printf("%lf ", val);
	}
	printf("\n");
    }
    
    return 0;
  }
