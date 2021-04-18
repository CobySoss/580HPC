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
#include <vector>
#include <algorithm>

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

void host_csr_spmm(CSR &mata, CSR &matb, CSR &matc) {    
   double* scatter = (double*)malloc(matc.ncols * sizeof(double));
   for(unsigned int i = 0; i < matc.ncols; i++)
   {
      scatter[i] = -1.0;
   }
   int numValsC = 0; 
   for(unsigned int r = 0; r < mata.nrows; r++)
   {
       unsigned int start_index_a = mata.row_indx[r];
       unsigned int end_index_a = mata.row_indx[r+1];
       for(unsigned int j = start_index_a; j < end_index_a; j++)
       {
          int colA = mata.col_id[j];
	  int valA = mata.values[j];
	  unsigned int start_index_b = matb.row_indx[colA];
	  unsigned int end_index_b = matb.row_indx[colA + 1];
	  for(unsigned int k = start_index_b; k < end_index_b; k++)
	  {
	     int colB =  matb.col_id[k];
	     int valB = matb.values[k];
	     if(scatter[colB] < 0.0)
	     {
		matc.col_id[numValsC] = colB;
		matc.values[numValsC] = valB * valA;
		scatter[colB] = numValsC;
		numValsC = numValsC + 1;
	     }
	     else
	     {
		int p = scatter[colB];
		matc.values[p] += valB * valA;
	     }
	  }
       }
       matc.row_indx[r + 1] = numValsC;
       for(unsigned int i = 0; i < matc.ncols; i++)
       {
         scatter[i] = -1.0;
       }
   }
   free(scatter);
}

__global__ void dev_csr_spgemm(CSR mata, CSR matb, CSR matc,  int* scatter, int* workload)
{
   int r = blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ int numValsC;
   if(threadIdx.x == 0)
   {
       numValsC = 0;
       matc.row_indx[0] = 0;
       for(unsigned int i = 0; i < matc.ncols; i++)
       {
          scatter[blockIdx.x * blockDim.x + i] = -1.0;
       }
   }
   __syncthreads();
   if(r < mata.nrows)
   {
        //r = workload[r];
   	unsigned int start_index_a = mata.row_indx[r];
   	unsigned int end_index_a = mata.row_indx[r+1];
   	for(unsigned int j = start_index_a; j < end_index_a; j++)
   	{
      	    numValsC = matc.row_indx[r];
            int colA = mata.col_id[j];
            int valA = mata.values[j];
            unsigned int start_index_b = matb.row_indx[colA];
            unsigned int end_index_b = matb.row_indx[colA + 1];
            int tempC = 0;
            for(unsigned int k = start_index_b; k < end_index_b; k++)
            {
                int colB =  matb.col_id[k];
                int valB = matb.values[k];
                if(scatter[blockIdx.x * blockDim.x + colB] < 0.0)
                {
                    int prevValsC = atomicAdd(&numValsC, 1);
                    matc.col_id[prevValsC] = colB;
                    tempC = valB * valA;
                    scatter[blockIdx.x * blockDim.x + colB] = prevValsC;
                }
                else
                {
                    tempC += valB * valA;
                    int p = scatter[blockIdx.x * blockDim.x + colB];
                    matc.values[p] = tempC;
                }
            } 
        }
        matc.row_indx[r + 1] = numValsC;
        for(unsigned int i = 0; i < matc.ncols; i++)
        {
            scatter[blockIdx.x * blockDim.x + i] = -1.0;
        }
   }   
}

std::tuple<int, int*> host_calc_sizec(CSR &mata, CSR &matb)
{
    std::vector<std::tuple<int, int>> rowsByWorkload;
    std::set<std::tuple<int, int>> row_column_in_c;
    for(unsigned int r = 0; r < mata.nrows; r++)
    {
	 int rowWork = 0;
         int starting_index = mata.row_indx[r];
         int ending_index = mata.row_indx[r + 1];
         for(unsigned int j = starting_index; j < ending_index; j++)
	 {
             int colA = mata.col_id[j];
	     int starting_index_b = matb.row_indx[colA];
	     int ending_index_b = matb.row_indx[colA + 1];
             for(unsigned int k = starting_index_b; k < ending_index_b; k++)
	     {
		rowWork++;
	     	int colB = matb.col_id[k];
		std::tuple<int, int> rcpair(r, colB);
		row_column_in_c.insert(rcpair);
	     }	     
	 }
         rowsByWorkload.push_back(std::tuple<int, int>(rowWork, r));	 
    }
    std::sort(rowsByWorkload.begin(), rowsByWorkload.end());
    int* reorderedRows = (int*)malloc(rowsByWorkload.size() * sizeof(int));
    for(unsigned int l = 0; l < rowsByWorkload.size(); l++)
    {
    	reorderedRows[l] = get<1>(rowsByWorkload[l]);
    }
    return std::tuple<int, int*>(row_column_in_c.size(), reorderedRows);
}

void test_spGEMM_densified_against_expected(double* a, double *b, unsigned int num ) {
        for (unsigned int k = 0; k < num; ++k) {
	    if(std::abs(a[num] - b[num]) > 1e-1) {
              std::cout << "error at index " << num << endl;
            }
        }
    }

void runCuda(CSR &mata, CSR &matb, int c_nnz, int* workload_row_order)
{
    int* d_scatter;
    int* d_workload_row_order;
    CSR d_mata, d_matb, d_matc;
    d_mata.nrows = mata.nrows;
    d_mata.ncols = mata.ncols;
    d_mata.nnz = mata.nnz;
    cudaMalloc(&d_mata.values, (mata.nnz * sizeof(double)));
    cudaMalloc(&d_mata.col_id, (mata.nnz * sizeof(int)));
    cudaMalloc(&d_mata.row_indx, (mata.nrows + 1) * sizeof(int));
    d_matb.nrows = matb.nrows;
    d_matb.ncols = matb.ncols;
    d_matb.nnz = matb.nnz;
    cudaMalloc(&d_matb.values, (matb.nnz * sizeof(double)));
    cudaMalloc(&d_matb.col_id, (matb.nnz * sizeof(int)));
    cudaMalloc(&d_matb.row_indx, (matb.nrows + 1) * sizeof(int));
    d_matc.nrows = mata.nrows;
    d_matc.ncols = matb.ncols;
    d_matc.nnz = c_nnz;
    cudaMalloc(&d_scatter, (matb.ncols * ceil(mata.nrows / 128.0) * sizeof(int)));
    cudaMalloc(&d_workload_row_order, mata.nrows * sizeof(int));
    cudaMalloc(&d_matc.values, (c_nnz * sizeof(double)));
    cudaMalloc(&d_matc.col_id, (c_nnz * sizeof(int)));
    cudaMalloc(&d_matc.row_indx, (mata.nrows + 1) * sizeof(int));
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaEvent_t streamOneMemcpyDone;
    cudaEvent_t streamTwoMemcpyDone;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);
    cudaEventCreate(&streamOneMemcpyDone);
    cudaEventCreate(&streamTwoMemcpyDone);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    double time_start = omp_get_wtime();
    cudaMemcpyAsync(d_mata.values, mata.values, (mata.nnz * sizeof(double)), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_mata.col_id, mata.col_id, (mata.nnz * sizeof(int)), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_mata.row_indx, mata.row_indx, (mata.nrows + 1) * sizeof(int), cudaMemcpyHostToDevice, stream1);
    
    cudaMemcpyAsync(d_matb.values, matb.values, (matb.nnz * sizeof(double)), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_matb.col_id, matb.col_id, (matb.nnz * sizeof(int)), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_matb.row_indx, matb.row_indx, (matb.nrows + 1) * sizeof(int), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_workload_row_order, workload_row_order, mata.nrows * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(streamOneMemcpyDone, stream1);
    cudaEventRecord(streamTwoMemcpyDone, stream2);
    dim3 block;
    block.x = 128;
    dim3 grid;
    grid.x = ceil(mata.nrows / 128.0);
    cudaEventSynchronize(streamOneMemcpyDone);
    cudaEventSynchronize(streamTwoMemcpyDone);
    cudaEventRecord(kernel_start, stream1);
    dev_csr_spgemm<<<grid, block, 0, stream1>>>(d_mata,d_matb, d_matc, d_scatter, d_workload_row_order);    
    cudaEventRecord(kernel_end, stream1);
    CSR matc;
    matc.nnz = d_matc.nnz;
    matc.nrows = d_matc.nrows;
    matc.ncols = d_matc.ncols;
    matc.values = (double*)malloc(d_matc.nnz * sizeof(double));
    matc.col_id = (unsigned int*)malloc(d_matc.nnz * sizeof(int));
    matc.row_indx = (unsigned int*)malloc((d_matc.nrows +1) * sizeof(int));
    cudaStreamSynchronize(stream1);
    float time;
    cudaEventElapsedTime(&time, kernel_start, kernel_end);
    printf("Kernel time: %f ms\n", time);
    cudaMemcpy(matc.values, d_matc.values, (d_matc.nnz * sizeof(double)), cudaMemcpyDeviceToHost);
    cudaMemcpy(matc.col_id, d_matc.col_id, (d_matc.nnz * sizeof(int)), cudaMemcpyDeviceToHost);
    cudaMemcpy(matc.row_indx, d_matc.row_indx, ((d_matc.nrows + 1) * sizeof(int)), cudaMemcpyDeviceToHost);
    double time_end = omp_get_wtime();
    printf("total time: %lf seconds\n", time_end - time_start);
    /*
    //test device spGEMM
   
    int nrowA = 0;
    int ncolA = 0;
    int nrowB = 0; 
    int ncolB = 0; 
    int nrowC = 0;
    int ncolC = 0;
    double* denseCResultFromSpGEMM = getDenseMat(matc, &nrowC, &ncolC);
    double* denseB = getDenseMat(matb, &nrowB, &ncolB);
    double* denseA = getDenseMat(mata, &nrowA, &nrowB);
    double* expectedMat = DenseDenseMult(denseA, nrowA, ncolA, denseB, nrowB, ncolB, &nrowC, &ncolC);
    test_spGEMM_densified_against_expected(expectedMat, denseCResultFromSpGEMM, nrowC * ncolC);
   */

    cudaFree(d_mata.values);
    cudaFree(d_mata.col_id);
    cudaFree(d_mata.row_indx);
   
    cudaFree(d_matb.values);
    cudaFree(d_matb.col_id);
    cudaFree(d_matb.row_indx);

   cudaFree(d_matc.values);
   cudaFree(d_matc.col_id);
   cudaFree(d_matc.row_indx);

}	
int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cerr << "usage ./exec inputfile K  " << std::endl;
        exit(-1);
    }

    CSR mata= read_matrix_market_to_CSR(argv[1]);
    CSR matb = read_matrix_market_to_CSR(argv[2]);
    CSR matc;
    //print_CSR(mat);
    std::cout << mata.nrows << ' ' << mata.ncols << ' ' << mata.nnz << ' ' << '\n';
    std::cout << matb.nrows << ' ' << matb.ncols << ' ' << matb.nnz << ' ' << '\n';

    std::tuple<int, int*> optInfo = host_calc_sizec(mata, matb);
    int nnz_c = get<0>(optInfo);
    matc.values = (double*)malloc(nnz_c * sizeof(double));
    matc.col_id = (unsigned int*)malloc(nnz_c * sizeof(double));
    matc.row_indx = (unsigned int*)malloc((mata.nrows + 1) * sizeof(double));
    matc.nrows = mata.nrows;
    matc.ncols = matb.ncols;
    matc.nnz = nnz_c;
    host_csr_spmm(mata, matb, matc);
    printf("nnz c %d\n", matc.nnz);
/*
    //test SpGEMM result against expected dense matrix mult -- HOST
    int nrowA = 0; 
    int ncolA = 0;
    int nrowB = 0;
    int ncolB = 0; 
    int nrowC = 0; 
    int ncolC = 0; 
    double* denseCResultFromSpGEMM = getDenseMat(matc, &nrowC, &ncolC);    
    double* denseB = getDenseMat(matb, &nrowB, &ncolB);
    double* denseA = getDenseMat(mata, &nrowA, &nrowB);
    double* expectedMat = DenseDenseMult(denseA, nrowA, ncolA, denseB, nrowB, ncolB, &nrowC, &ncolC);
    test_spGEMM_densified_against_expected(expectedMat, denseCResultFromSpGEMM, nrowC * ncolC); 
  */  
    runCuda(mata, matb, nnz_c, get<1>(optInfo));
    free(get<1>(optInfo));
    //free(denseCResultFromSpGEMM);
    //free(denseA);
    //free(denseB);
    free(mata.row_indx);
    free(mata.col_id);
    free(mata.values);

    

    
    return 0;
}
