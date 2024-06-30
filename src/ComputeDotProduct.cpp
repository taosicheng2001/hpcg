
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "mytimer.hpp"
#include <mpi.h>
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include <immintrin.h>

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication
  between processes
  @param[out] isOptimized should be set to false if this routine uses the
  reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector &x, const Vector &y,
                      double &result, double &time_allreduce,
                      bool &isOptimized) {

  // This line and the next two lines should be removed and your version of
  // ComputeDotProduct should be used.
  // isOptimized = false;
  // return ComputeDotProduct_ref(n, x, y, result, time_allreduce);

  assert(x.localLength >= n);
  assert(y.localLength >= n);

  double local_result = 0.0;
  double *xv = x.values;
  double *yv = y.values;

#ifndef HPCG_NO_OPENMP
#pragma omp declare reduction(+ : __m512d : omp_out = omp_out + omp_in) \
    initializer(omp_priv = _mm512_setzero_pd())
#endif

  __m512d sum_vec = _mm512_setzero_pd();

  if (yv == xv) {

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : sum_vec)
#endif
    for (local_int_t i = 0; i <= n - 8; i += 8) {
      __m512d x0 = _mm512_loadu_pd(&xv[i]);
      // sum_vec += x0 * x0;
      sum_vec = _mm512_fmadd_pd(x0, x0, sum_vec);
    }
    local_result = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
    
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : local_result)
#endif
    for (local_int_t res_i = n / 8 * 8; res_i < n; res_i++) {
      local_result += xv[res_i] * xv[res_i];
    }
  } else {

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : sum_vec)
#endif
    for (local_int_t i = 0; i <= n - 8; i += 8) {
      __m512d x0 = _mm512_loadu_pd(&xv[i]);
      __m512d y0 = _mm512_loadu_pd(&yv[i]);
      sum_vec += _mm512_fmadd_pd(x0, y0, sum_vec);
    }

    local_result = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : local_result)
#endif
    for (local_int_t res_i = n / 8 * 8; res_i < n; res_i++) {
      local_result += xv[res_i] * yv[res_i];
    }
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
