
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
  @file OptimizeProblem.cpp

  HPCG routine
  */

#include "OptimizeProblem.hpp"

#include <iostream>

/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
  */
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints

#if defined(HPCG_USE_MULTICOLORING)
  const local_int_t nrow = A.localNumberOfRows;
  std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  int totalColors = 1;
  colors[0] = 0; // first point gets color 0

  // Finds colors in a greedy (a likely non-optimal) fashion.

  for (local_int_t i=1; i < nrow; ++i) {
    if (colors[i] == nrow) { // if color not assigned
      std::vector<int> assigned(totalColors, 0);
      int currentlyAssigned = 0;
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
        local_int_t curCol = currentColIndices[j];
        if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
          if (assigned[colors[curCol]] == 0)
            currentlyAssigned += 1;
          assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
        } // else // could take advantage of indices being sorted
      }

      if (currentlyAssigned < totalColors) { // if there is at least one color left to use
        for (int j=0; j < totalColors; ++j)  // try all current colors
          if (assigned[j] == 0) { // if no neighbor with this color
            colors[i] = j;
            break;
          }
      } else {
        if (colors[i] == nrow) {
          colors[i] = totalColors;
          totalColors += 1;
        }
      }
    }
  }


  /* colors[i] save the color of row "i" */
  /* counters[i] save the number of rows assigned to color "i" */
  std::vector<local_int_t> counters(totalColors);
  for (local_int_t i=0; i<nrow; ++i)
    counters[colors[i]]++;

  /* colorIndices save the first row with color "i" */
  /* computationOrder save the order that used in SYMGS */
  std::vector<local_int_t> colorIndices(totalColors, 0);
  local_int_t old = 0, old0;
  for ( int i = 0; i < totalColors; i++ ) {
  	old0 = counters[i];
  	counters[i] = counters[i-1] + old;
  	colorIndices[i] = counters[i];
  	old = old0;
  }
  counters[0] = 0;
  colorIndices[0] = 0;

  std::vector<local_int_t> computationOrder(nrow);
  local_int_t j = 0;
  for ( int ic = 0; ic < colorIndices.size(); ic++ ) {
  	for ( local_int_t i = 0; i < nrow; i++ ) {
  		if ( colors[i] == ic ) {
  			computationOrder[j] = i;
  			j++;
  		}
  	}
  }

  /* save the metadata to Matrix */
  A.optimizationData[0] = std::vector<local_int_t>(computationOrder);
  A.optimizationData[1] = std::vector<local_int_t>(colorIndices);


#ifdef HPCG_USE_REORDER_MULTICOLORING

  /* Here we obtain colors of each row */
  /* We reorder the row with same color */

  // allocate some structure to temporary allocate reordered structures
  double **matrixValues = new double*[nrow];
  local_int_t **mtxIndL = new local_int_t*[nrow];
  char *nonzerosInRow = new char[nrow];
  for (local_int_t i = 0; i < nrow; i++){
	matrixValues[i] = new double[27];
	mtxIndL[i] = new local_int_t[27];
  }

  // reorder and translate
  Vector bReorder;
  InitializeVector(bReorder, b.localLength);

  local_int_t numberOfReorderedRow = 0;
  for(local_int_t c = 0; c < totalColors; c++){
	for(local_int_t i = 0; i < nrow; i++){
		if(colors[i] == c){ // select the row "i" with color "c"
			nonzerosInRow[numberOfReorderedRow] = A.nonzerosInRow[i];
			bReorder.values[numberOfReorderedRow] = b.values[i];

			for(local_int_t j = 0; j < A.nonzerosInRow[i]; j++){
				local_int_t curOldCol = A.mtxIndL[i][j];
				matrixValues[numberOfReorderedRow][j] = A.matrixValues[i][j];
				mtxIndL[numberOfReorderedRow][j] = curOldCol;
			}
			numberOfReorderedRow++; // step to next row
		}
	}
  }

  // replace structure
  for(local_int_t i = 0; i < nrow; i++){
	A.nonzerosInRow[i] = nonzerosInRow[i];
	for(local_int_t j = 0; j < A.nonzerosInRow[i]; j++){
		A.matrixValues[i][j] = matrixValues[i][j];
		A.mtxIndL[i][j] = mtxIndL[i][j];
	}
	for(local_int_t j = A.nonzerosInRow[i]; j < 27; j++){
		A.matrixValues[i][j] = 0.0;
		A.mtxIndL[i][j] = 0;
	}
	for(local_int_t j = 0; j < b.localLength; j++){
		b.values[j] = bReorder.values[j];
	}
  }

  // regenerate diagonal
  for(local_int_t i = 0; i < nrow; i++){
	for(local_int_t j = 0; j < A.nonzerosInRow[i]; j++){
		local_int_t curCol = A.mtxIndL[i][j];
		if( i == curCol)
			A.matrixDiagonal[i] = &A.matrixValues[i][j];
	for(local_int_t j = A.nonzerosInRow[i]; j < 27; j++){
			A.matrixValues[i][j] = 0.0;
			A.mtxIndL[i][j] = 0;
		}
	}

  }

#endif

#endif

	// Reorder b (RHS) vector
	Vector bReorder;
	InitializeVector(bReorder, b.localLength);
	CopyVector(b, bReorder);
	CopyAndReorderVector(bReorder, b, A.whichNewRowIsOldRow);

#ifdef HPCG_USE_SPMV_ARMPL
	// Create the temporary data structures that will be copied (flags=0) inside the create call
	local_int_t m = A.localNumberOfRows;
	local_int_t n = A.localNumberOfColumns;
	
	local_int_t nnz = A.localNumberOfNonzeros;

	armpl_int_t *row_ptr = (armpl_int_t*) std::malloc((m+1)*sizeof(armpl_int_t));
	armpl_int_t *col_idx = (armpl_int_t*) std::malloc(nnz*sizeof(armpl_int_t));
	double *vals = (double*) std::malloc(nnz*sizeof(double));

	row_ptr[0] = 0;
	for ( local_int_t i = 0; i < m; i++ ) {
		row_ptr[i+1] = row_ptr[i] + A.nonzerosInRow[i];
	}

	global_int_t k = 0;
	for ( local_int_t i = 0; i < m; i++ ) {
		for ( local_int_t j = 0; j < A.nonzerosInRow[i]; j++ ) {
			col_idx[k] = A.mtxIndL[i][j];
			vals[k++] = A.matrixValues[i][j];
		}
	}

	armpl_int_t flags = 0;
	armpl_spmat_create_csr_d(&A.armpl_mat, m, n, row_ptr, col_idx, vals, flags);

	free(row_ptr);
	free(col_idx);
	free(vals);

	armpl_spmat_hint(A.armpl_mat, ARMPL_SPARSE_HINT_STRUCTURE, ARMPL_SPARSE_STRUCTURE_HPCG);
	armpl_spmv_optimize(A.armpl_mat);
#endif

	if ( A.mgData != 0 ) {
		// Translate f2cOperator values
		local_int_t ncrow = (A.geom->nx/2) * (A.geom->ny/2) * (A.geom->nz/2);
		for ( local_int_t i = 0; i < ncrow; i++ ) {
			local_int_t orig = A.mgData->f2cOperator[i];
			A.mgData->f2cOperator[i] = A.whichNewRowIsOldRow[orig];
		}

		return OptimizeCoarseProblem(*A.Ac);
	}

	return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

	return 0.0;

}
