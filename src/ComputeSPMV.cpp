
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV_ref.hpp"
#include<immintrin.h>
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/


int ComputeSPMV(const SparseMatrix &A, Vector &x, Vector &y)
{

  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  A.isSpmvOptimized = true;
  //return ComputeSPMV_ref(A,x,y);
  assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
  for (local_int_t i = 0; i < nrow; i++) {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    int j = 0;

    __m256d vec_sum = _mm256_setzero_pd(); // Initialize sum vector

    for (; j <= cur_nnz - 4; j += 4) {
      __m256d vec_vals = _mm256_load_pd(&cur_vals[j]); // Aligned load of 4 values
      __m256d vec_xv = _mm256_set_pd(xv[cur_inds[j+3]], xv[cur_inds[j+2]], xv[cur_inds[j+1]], xv[cur_inds[j]]);
      vec_sum = _mm256_fmadd_pd(vec_vals, vec_xv, vec_sum); // Fused multiply-add
    }

    // Horizontal add to get the final sum from the vector
    vec_sum = _mm256_hadd_pd(vec_sum, vec_sum);
    sum = ((double*)&vec_sum)[0] + ((double*)&vec_sum)[2];

    // Process remaining elements
    for (; j < cur_nnz; j++) {
      sum += cur_vals[j] * xv[cur_inds[j]];
    }

    yv[i] = sum;
  }
return 0;
}
