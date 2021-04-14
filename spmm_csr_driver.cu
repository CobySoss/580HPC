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
#include  <omp.h>
#include <tuple>
#include <set>

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

void host_csr_spmm(CSR &mat, CSR &matb, CSR &matc) {
    
}

__global__ void dev_csr_spmm(CSR mata, CSR matb, CSR matc,  unsigned int offset)
{
	int row = offset + blockIdx.x * blockDim.x + threadIdx.x;
	
}

int host_calc_sizec(CSR &mata, CSR &matb)
{
    int sizec = 0;
    std::set<std::tuple<int, int>> row_column_in_c;
    for(unsigned int r = 0; r < mata.nrows; r++)
    {
         int starting_index = mata.row_indx[r];
         int ending_index = mata.row_indx[r + 1];
         for(unsigned int j = starting_index; j < ending_index; j++)
	 {
             int colA = mata.col_id[j];
	     int starting_index_b = matb.row_indx[colA];
	     int ending_index_b = matb.row_indx[colA + 1];
             for(unsigned int k = starting_index_b; k < ending_index_b; k++)
	     {
	
	     	int colB = matb.col_id[k];
		std::tuple<int, int> rcpair(r, colB);
		printf("%d %d\n", r, colB);
		row_column_in_c.insert(rcpair);
	     }	     
	 }	 
    }
    return row_column_in_c.size();
}

int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cerr << "usage ./exec inputfile K  " << std::endl;
        exit(-1);
    }

    CSR mat = read_matrix_market_to_CSR(argv[1]);
    CSR matb = read_matrix_market_to_CSR(argv[2]);
    CSR matc;
    //print_CSR(mat);
    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << ' ' << '\n';
    std::cout << mat.nrows << ' ' << matb.ncols << ' ' << mat.nnz << ' ' << '\n';

    int nnz_c = host_calc_sizec(mat, matb);
    printf("how big %d\n", nnz_c);
    
    host_csr_spmm(mat, matb, matc);
     
    
   
    free(mat.row_indx);
    free(mat.col_id);
    free(mat.values);
    
    return 0;
  }
