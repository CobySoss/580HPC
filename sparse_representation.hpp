/*
 * sparse_representation.hpp
 * Copyright (C) 2020 
 * 	Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#ifndef SPARSE_REPRESENTATION_HPP
#define SPARSE_REPRESENTATION_HPP

struct CSR
{
	unsigned int* row_indx{};
    unsigned int* col_id{}; 	
	double* values{};

	unsigned int nrows{};
	unsigned int ncols{};
	unsigned int nnz{};
};

struct CSC
{
    unsigned int* col_indx{}; 	
	unsigned int* row_id{};
	double* values{};

	unsigned int nrows{};
	unsigned int ncols{};
	unsigned int nnz{};
};

struct COO
{
	unsigned int* row_id{};
    unsigned int* col_id{}; 	
	double* values{};

	unsigned int nrows{};
	unsigned int ncols{};
	unsigned int nnz{};
};

#endif /* !SPARSE_REPRESENTATION_HPP */

