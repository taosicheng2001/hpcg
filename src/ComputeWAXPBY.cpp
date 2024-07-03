
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

#include <immintrin.h>

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector &x,
                  const double beta, const Vector &y, Vector &w, bool &isOptimized)
{

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  isOptimized = true;
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  const double *const xv = x.values;
  const double *const yv = y.values;
  double *const wv = w.values;

    local_int_t n_vec = n / 4 * 4; // 向量化长度的倍数部分
    local_int_t i;

    if (alpha == 1.0) {
//#pragma omp parallel for 
        for (i = 0; i < n_vec; i += 4) {
            __m256d beta_vec = _mm256_set1_pd(beta);
            __m256d xv_vec = _mm256_load_pd(xv + i);
            __m256d yv_vec = _mm256_load_pd(yv + i);
            //__m256d wv_vec = _mm256_add_pd(xv_vec, _mm256_mul_pd(beta_vec, yv_vec));
            __m256d wv_vec = _mm256_fmadd_pd(beta_vec, yv_vec, xv_vec);
            _mm256_storeu_pd(wv + i, wv_vec);
        }
//#pragma omp parallel for 
        for (; i < n; ++i) {
            wv[i] = xv[i] + beta * yv[i];
        }
    } else if (beta == 1.0) {
//#pragma omp parallel for 
        for (i = 0; i < n_vec; i += 4) {
            __m256d alpha_vec = _mm256_set1_pd(alpha);
            __m256d xv_vec = _mm256_load_pd(xv + i);
            __m256d yv_vec = _mm256_load_pd(yv + i);
            //__m256d wv_vec = _mm256_add_pd(yv_vec, _mm256_mul_pd(alpha_vec, xv_vec));
            __m256d wv_vec = _mm256_fmadd_pd(alpha_vec, xv_vec, yv_vec);
            _mm256_storeu_pd(wv + i, wv_vec);
        }
//#pragma omp parallel for 
        for (; i < n; ++i) {
            wv[i] = yv[i] + alpha * xv[i];
        }
    } else {
//#pragma omp parallel for 
        for (i = 0; i < n_vec; i += 4) {
            __m256d alpha_vec = _mm256_set1_pd(alpha);
            __m256d beta_vec = _mm256_set1_pd(beta);
            __m256d xv_vec = _mm256_load_pd(xv + i);
            __m256d yv_vec = _mm256_load_pd(yv + i);
            __m256d wv_vec = _mm256_fmadd_pd(alpha_vec, xv_vec, _mm256_mul_pd(beta_vec, yv_vec)); // w = alpha * x + beta * y
            _mm256_storeu_pd(wv + i, wv_vec);
        }
//#pragma omp parallel for 
        for (; i < n; ++i) {
            wv[i] = alpha * xv[i] + beta * yv[i];
        }
    }
  return 0;
}
