/*
 * mm_helper.cpp
 * Copyright (C) 2020
 *  Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#include "mm_helper.hpp"
#include "sparse_representation.hpp"
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <assert.h>
#include <tuple>
#include <vector>
using namespace std;
COO read_matrix_market_to_COO(const char* fname) {
    COO mat;
    std::ifstream f(fname);

    while(f.peek() == '%') {
        f.ignore(2048, '\n');
    }

    f >> mat.nrows >> mat.ncols >> mat.nnz;
    //std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << std::endl;

    mat.row_id = (unsigned int*) malloc (mat.nnz * sizeof(unsigned int));
    mat.col_id = (unsigned int*) malloc (mat.nnz * sizeof(unsigned int));
    mat.values = (double*) malloc (mat.nnz * sizeof(double));

    for(unsigned int i = 0; i < mat.nnz; i++) {
        unsigned int m;
        unsigned int n;
        double val;
        f >> m >> n >> val;
        //std::cout << m << ' ' << n << ' ' << val << std::endl;
        mat.row_id[i] = --m;
        mat.col_id[i] = --n;
        mat.values[i] = val;
    }

    return mat;
}

COO** getSameColumnMatrixFragments(COO mat, int* numCOO)
{
   std::vector<tuple<int, int, double>> cooEntries;
   std::vector<tuple<int, int, double>> matrixFragment;
   std::vector<COO*> result;
   for(unsigned int i = 0; i < mat.nnz; i++)
   {
      cooEntries.push_back(std::make_tuple(mat.col_id[i], mat.row_id[i], mat.values[i]));
   }
   std::sort(cooEntries.begin(), cooEntries.end());
   int prevColumn = -1;
   COO* currentCOO;
   int maxFragmentSize = -1;
   for(int i = 0; i < cooEntries.size(); i++)
   {
       if(prevColumn != get<0>(cooEntries[i]) && prevColumn != -1)
       {
          COO* fragment = (COO*)malloc(sizeof(COO));
          fragment->nnz = matrixFragment.size();
	  fragment->values = (double*)malloc(matrixFragment.size() * sizeof(double));
	  fragment->row_id = (unsigned int*)malloc(matrixFragment.size() * sizeof(int));
	  fragment->col_id = (unsigned int*)malloc(matrixFragment.size() * sizeof(int));
	  for(int j = 0; j < matrixFragment.size(); j++)
	  {
	     fragment->values[j] = get<2>(matrixFragment[j]);
	     fragment->col_id[j] = get<0>(matrixFragment[j]);
	     fragment->row_id[j] = get<1>(matrixFragment[j]);
	  }
	  matrixFragment.clear();
	  result.push_back(fragment);
       }
   	//cout << get<0>(cooEntries[i]) << " " 
        //     << get<1>(cooEntries[i]) << " "
        //     << get<2>(cooEntries[i]) << "\n";
	matrixFragment.push_back(cooEntries[i]);
	prevColumn = get<0>(cooEntries[i]);
   }
   if(matrixFragment.size() != 0)
   {
       COO* fragment = (COO*)malloc(sizeof(COO));
       fragment->nnz = matrixFragment.size();
       fragment->values = (double*)malloc(matrixFragment.size() * sizeof(double));
       fragment->row_id = (unsigned int*)malloc(matrixFragment.size() * sizeof(int));
       fragment->col_id = (unsigned int*)malloc(matrixFragment.size() * sizeof(int));
       for(int j = 0; j < matrixFragment.size(); j++)
       {
          fragment->values[j] = get<2>(matrixFragment[j]);
          fragment->col_id[j] = get<0>(matrixFragment[j]);
          fragment->row_id[j] = get<1>(matrixFragment[j]);
       }
        matrixFragment.clear();
        result.push_back(fragment);
   }
   (*numCOO) = result.size();
   COO** a = (COO**)malloc(result.size() * sizeof(COO*));
   for(unsigned int k = 0; k < result.size(); k++)
   {
   	a[k] = result[k];
   }
   return a;
}

CSR read_matrix_market_to_CSR(const char* fname) {
    COO coo_mat  = read_matrix_market_to_COO(fname);
    unsigned int* idx_arr = (unsigned int*) malloc (coo_mat.nnz * sizeof(unsigned int));

    std::iota(idx_arr, idx_arr + coo_mat.nnz, 0 );
    std::sort(idx_arr, idx_arr + coo_mat.nnz, [&coo_mat](unsigned int i, unsigned int j) {
        if(coo_mat.row_id[i] < coo_mat.row_id[j])
            return true;
        else if(coo_mat.row_id[i] > coo_mat.row_id[j])
            return false;
        else if(coo_mat.col_id[i] < coo_mat.col_id[j])
            return true;
        else
            return false;
    });

    CSR csr_mat;
    csr_mat.nnz = coo_mat.nnz;
    csr_mat.nrows = coo_mat.nrows;
    csr_mat.ncols = coo_mat.ncols;

    csr_mat.row_indx = (unsigned int*) malloc ((csr_mat.nrows + 1) * sizeof(unsigned int));
    csr_mat.col_id = (unsigned int*) malloc (csr_mat.nnz * sizeof(unsigned int));
    csr_mat.values = (double*) malloc (csr_mat.nnz * sizeof(double));

    unsigned int prev_row = 0;
    int cnt = 0;
    csr_mat.row_indx[0] = 0;

    for (unsigned int i = 0; i < csr_mat.nnz; ++i) {
        auto cur_idx  = idx_arr[i];
        auto cur_row = coo_mat.row_id[cur_idx];
        assert(prev_row <= cur_row);
        while(prev_row != cur_row) {
            csr_mat.row_indx[prev_row + 1] = csr_mat.row_indx[prev_row] + cnt;
            cnt = 0;
            prev_row++;
        }
        cnt++;

        csr_mat.col_id[i] = coo_mat.col_id[cur_idx];
        csr_mat.values[i] = coo_mat.values[cur_idx];
    }
    while(prev_row < csr_mat.nrows ) {
        csr_mat.row_indx[prev_row + 1] = csr_mat.row_indx[prev_row] + cnt;
        cnt = 0;
        prev_row++;
    }

    free(coo_mat.row_id);
    free(coo_mat.col_id);
    free(coo_mat.values);

    return csr_mat;


}
double * getDenseMat(CSR mat, int* nrow, int* ncol)
{
    double* denseMat = (double*)malloc(mat.nrows * mat.ncols * sizeof(double));   
    for(unsigned int m = 0; m < mat.nrows; m++)
    {
        for(unsigned int n = 0; n < mat.ncols; n++)
        {
            denseMat[m * mat.ncols + n] = 0.0;
        }
   }
   for(unsigned int r = 0; r < mat.nrows; r++)
   {
      int startIndex = mat.row_indx[r];
      int endIndex = mat.row_indx[r+1];
      for(unsigned int j = startIndex; j < endIndex; j++)
      {
          int colid = mat.col_id[j];
	  double val = mat.values[j];
	  denseMat[r * mat.ncols + colid] = val;	  
      }
   }
   (*nrow) = mat.nrows;
   (*ncol) = mat.ncols;
   return denseMat;
}

double * DenseDenseMult(double* A, int rowA, int colA, double* B, int rowB, int colB, int* rowC, int* colC)
{
   double* C = (double*)malloc(rowA * colB * sizeof(double));
   for(int r = 0; r < rowA; r++)
   {
   	for(int c = 0; c < colB; c++)
	{
	    C[r * colB + c] = 0.0;
	    for(int ca = 0; ca < colA; ca++)
	    {
	        C[r * colB + c] += A[r * colA + ca] * B[ca * colB +c];
	    }
	}
   }
   (*rowC) = rowA;
   (*colC) = colB;
   return C;
}

double * getDenseMat(std::string filename, int* nrow, int* ncol)
{
   ifstream file;
   file.open(filename);
   int NumRow, NumCol, NNZ;
   std::string header;
   file >> header;
   file >> NumRow >> NumCol  >> NNZ;
   
   cout << NumRow << " " << NumCol << " " << NNZ << endl;
   double* mat = (double*)malloc(NumRow * NumCol * sizeof(double));
   for(unsigned int m = 0; m < NumRow; m++)
   {
      for(unsigned int n = 0; n < NumCol; n++)
      {
          mat[m * NumCol + n] = 0.0;
      } 
   }
   for(unsigned int nz = 0; nz < NNZ; nz++)
   {
      int row, col;
      double val;
      file >> row >> col >> val;
      mat[(row-1) * NumCol + (col-1)] = val;
   }
  /*
   for(unsigned int m = 0; m < NumRow; m++)
   {
      for(unsigned int n = 0; n < NumCol; n++)
      {
          if(mat[m * NumCol + n] - -1.0 > .001)
	  {
	  	cout << m+1 << " " << n + 1 << " " << mat[m * NumCol +n] <<" " << endl;
	  }
      }
   }
   */
   file.close();
   return mat;
}

CSC read_matrix_market_to_CSC(const char* fname) {
    COO coo_mat  = read_matrix_market_to_COO(fname);
    unsigned int* idx_arr = (unsigned int*) malloc (coo_mat.nnz * sizeof(unsigned int));

    std::iota(idx_arr, idx_arr + coo_mat.nnz, 0 );
    std::sort(idx_arr, idx_arr + coo_mat.nnz, [&coo_mat](unsigned int i, unsigned int j) {
        if(coo_mat.col_id[i] < coo_mat.col_id[j])
            return true;
        else if(coo_mat.col_id[i] > coo_mat.col_id[j])
            return false;
        else if(coo_mat.row_id[i] < coo_mat.row_id[j])
            return true;
        else
            return false;
    });

    CSC csc_mat;
    csc_mat.nnz = coo_mat.nnz;
    csc_mat.nrows = coo_mat.nrows;
    csc_mat.ncols = coo_mat.ncols;

    csc_mat.col_indx = (unsigned int*) malloc ((csc_mat.ncols + 1) * sizeof(unsigned int));
    csc_mat.row_id = (unsigned int*) malloc (csc_mat.nnz * sizeof(unsigned int));
    csc_mat.values = (double*) malloc (csc_mat.nnz * sizeof(double));

    unsigned int prev_col = 0;
    int cnt = 0;
    csc_mat.col_indx[0] = 0;

    for (unsigned int i = 0; i < csc_mat.nnz; ++i) {
        auto cur_idx  = idx_arr[i];
        auto cur_col = coo_mat.col_id[cur_idx];
        assert(prev_col <= cur_col);
        while(prev_col != cur_col) {
            csc_mat.col_indx[prev_col + 1] = csc_mat.col_indx[prev_col] + cnt;
            cnt = 0;
            prev_col++;
        }
        cnt++;

        csc_mat.row_id[i] = coo_mat.row_id[cur_idx];
        csc_mat.values[i] = coo_mat.values[cur_idx];
    }
    while(prev_col < csc_mat.ncols ) {
        csc_mat.col_indx[prev_col + 1] = csc_mat.col_indx[prev_col] + cnt;
        cnt = 0;
        prev_col++;
    }

    free(coo_mat.col_id);
    free(coo_mat.row_id);
    free(coo_mat.values);

    return csc_mat;

}

