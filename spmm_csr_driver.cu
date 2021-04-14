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

__global__ void dev_csr_spmm(CSR mat, double* mat_in, double* mat_out, unsigned int K, unsigned int offset)
{
	int row = offset + blockIdx.x * blockDim.x + threadIdx.x;
	//printf("offset %d\n", offset);
	if(row < mat.nrows)
	{
	   unsigned int row_start = mat.row_indx[row];
	   unsigned int row_end = mat.row_indx[row + 1];
	   //printf("%d\n", row_end - row_start);
	   for(unsigned int k = 0; k < K; k++)
	   {
	   	mat_out[row * K + k] = 0;
	   }
	   //printf("%d %d %d\n", row, row_start, row_end);
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
        std::cerr << "usage ./exec inputfile K  " << std::endl;
        exit(-1);
    }

    unsigned int K = std::atoi(argv[2]);
    CSR mat = read_matrix_market_to_CSR(argv[1]);
    //print_CSR(mat);
    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << ' ' << K << '\n';

    double *dmat_in = (double*)malloc(mat.ncols * K  * sizeof(double));
    double *dmat_out = (double*)malloc(mat.nrows * K * sizeof(double));
    double *d_result_out = (double*)malloc(mat.nrows * K * sizeof(double));

    init_dmat(dmat_in, mat.ncols, K,  1.0);
    //print_dmat(dmat_in, mat.ncols, K);

    host_csr_spmm(mat, dmat_in, dmat_out, K);
     
    //allocate sparse matrix for device
    CSR d_mat;
    d_mat.ncols = mat.ncols;
    d_mat.nrows = mat.nrows;
    d_mat.nnz = mat.nnz;

    double *d_dmat_in;
    double *d_dmat_out;
    cudaMalloc(&d_dmat_in, mat.ncols * K * sizeof(double));
    cudaMalloc(&d_dmat_out, mat.nrows * K * sizeof(double));
    cudaMalloc(&d_mat.row_indx, (mat.nrows + 1) * sizeof(int));
    cudaMalloc(&d_mat.col_id, mat.nnz * sizeof(int));
    cudaMalloc(&d_mat.values, mat.nnz * sizeof(double));
    cudaMemcpy(d_dmat_in, dmat_in, mat.ncols * K * sizeof(double), cudaMemcpyHostToDevice);

    cudaStream_t cudaStreams[16];
    for(unsigned int s = 0; s < 16; s++)
    {
        cudaStreamCreate(&cudaStreams[s]);
    }
    int chunksize = 200;
    int chunkNum = 0;
    cudaEvent_t** cudaEvents = (cudaEvent_t**)malloc(sizeof(cudaEvent_t*) * (mat.nrows/chunksize) * 2);
    for(unsigned int e = 0; e < (mat.nrows/chunksize) * 2; e++)
    {
        cudaEvents[e] = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
        cudaEventCreate(cudaEvents[e]);
    }
    cudaEvent_t lastEventStart, lastEventEnd;
    cudaEventCreate(&lastEventStart);
    cudaEventCreate(&lastEventEnd);
    double start_time = omp_get_wtime();
    int event = 0;
    for(unsigned int c = 0; c < mat.nrows; c+= chunksize)
    {
	if(mat.nrows - c >= chunksize)
	{
           int stream = chunkNum % 16;
           //printf("chunksize %d\n", mat.row_indx[c + chunksize] - mat.row_indx[c]);
           cudaMemcpyAsync(d_mat.values + mat.row_indx[c], mat.values + mat.row_indx[c], (mat.row_indx[c + chunksize] - mat.row_indx[c]) * sizeof(double), cudaMemcpyHostToDevice, cudaStreams[stream]);
           cudaMemcpyAsync(d_mat.col_id + mat.row_indx[c], mat.col_id + mat.row_indx[c], (mat.row_indx[c + chunksize] - mat.row_indx[c]) * sizeof(int), cudaMemcpyHostToDevice, cudaStreams[stream]);
           cudaMemcpyAsync(d_mat.row_indx + c, mat.row_indx + c, (chunksize + 1)* sizeof(int), cudaMemcpyHostToDevice, cudaStreams[stream]);     
           dim3 block;
           block.x = chunksize;
           dim3 grid;
           grid.x = 1;
           //printf("Calling one kernel\n");
           cudaEventRecord(*cudaEvents[event]);	   
           dev_csr_spmm<<<grid, block, 0, cudaStreams[stream]>>>(d_mat, d_dmat_in, d_dmat_out, K, c);
           cudaEventRecord(*cudaEvents[event + 1]);
           event++;
	}

    }
    int remainder = mat.nrows % chunksize;
    if(remainder > 0)
    {
        //printf("remainder chunksize %d\n", mat.row_indx[mat.nrows] - mat.row_indx[mat.nrows - remainder]);
        cudaMemcpyAsync(d_mat.values + mat.row_indx[mat.nrows-remainder], 
			mat.values + mat.row_indx[mat.nrows - remainder], 
			(mat.row_indx[mat.nrows] - mat.row_indx[mat.nrows - remainder]) * sizeof(double), 
			cudaMemcpyHostToDevice, 
			cudaStreams[0]);
        cudaMemcpyAsync(d_mat.col_id + mat.row_indx[mat.nrows - remainder], 
			mat.col_id + mat.row_indx[mat.nrows - remainder], 
			(mat.row_indx[mat.nrows] - mat.row_indx[mat.nrows - remainder]) * sizeof(int), 
			cudaMemcpyHostToDevice, 
			cudaStreams[0]);
        cudaMemcpyAsync(d_mat.row_indx + mat.nrows - remainder, 
			mat.row_indx + mat.nrows - remainder, 
			(remainder + 1)* sizeof(int), 
			cudaMemcpyHostToDevice, 
			cudaStreams[0]);
	
        dim3 block;
        block.x = chunksize;
        dim3 grid;
        grid.x = 1;
        //printf("Calling one kernel\n");
	//printf("remainder %d\n", remainder);
	cudaEventRecord(lastEventStart);
        dev_csr_spmm<<<grid, block, 0, cudaStreams[0]>>>(d_mat, d_dmat_in, d_dmat_out, K, mat.nrows - remainder);
        cudaEventRecord(lastEventEnd);
    }
    for(unsigned int s = 0; s < 16; s++)
    {
    	cudaStreamSynchronize(cudaStreams[s]);
	cudaStreamDestroy(cudaStreams[s]);
    }
    cudaMemcpy(d_result_out, d_dmat_out, (mat.nrows * K * sizeof(double)), cudaMemcpyDeviceToHost);
    double end_time = omp_get_wtime();
    printf("Total time: %lf seconds \n", end_time - start_time);
    check_dmat(dmat_out, d_result_out, mat.nrows, K);
    //print_dmat(dmat_out, mat.nrows, K);
    float totalKernelTime = 0;
    for(unsigned int e = 0; e < (mat.nrows/chunksize); e++)
    {
        float time;
	cudaEventElapsedTime(&time, *cudaEvents[e * 2], *cudaEvents[e * 2 + 1]);
        cudaEventDestroy(*cudaEvents[e * 2]);
	free(cudaEvents[e * 2]);
	cudaEventDestroy(*cudaEvents[e * 2 + 1]);
	free(cudaEvents[e * 2+ 1]);
	totalKernelTime += time;
    }
    if(remainder > 0)
    {
	float lastTime = 0;
    	cudaEventElapsedTime(&lastTime, lastEventStart, lastEventEnd);
	totalKernelTime+=lastTime;
	cudaEventDestroy(lastEventStart);
	cudaEventDestroy(lastEventEnd);
    }
    printf("kernel time: %lf\n", totalKernelTime);
    free(cudaEvents);
    free(mat.row_indx);
    free(mat.col_id);
    free(mat.values);
    free(d_result_out);
    cudaFree(d_mat.row_indx);
    cudaFree(d_mat.col_id);
    cudaFree(d_mat.values);
    cudaFree(d_dmat_in);
    cudaFree(d_dmat_out); 
    return 0;
  }
