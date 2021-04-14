/*
 * mm_helper.hpp
 * Copyright (C) 2020 
 * 	Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#ifndef MM_HELPER_HPP
#define MM_HELPER_HPP

#include "sparse_representation.hpp"
#include <string>
using namespace std;
COO read_matrix_market_to_COO(const char* fname);
CSR read_matrix_market_to_CSR(const char* fname);
CSC read_matrix_market_to_CSC(const char* fname);
COO** getSameColumnMatrixFragments(COO mat, int* numCOO);
double * getDenseMat(CSR mat, int* nrow, int* ncol);
double * getDenseMat(std::string file, int* nrow, int* ncol);
double * DenseDenseMult(double* A, int rowA, int colA, double* B, int rowB, int colB, int* rowC, int* colC);
#endif /* !MM_HELPER_HPP */
