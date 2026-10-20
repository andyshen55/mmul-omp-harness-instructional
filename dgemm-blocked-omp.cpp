#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include "likwid-stuff.h"

const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

   // std::cout << "Insert your blocked matrix multiply with copy optimization, openmp-parallel edition here " << std::endl;

   const int b = block_size;
   const std::size_t tile_size = static_cast<std::size_t>(b) * b;

#pragma omp parallel
   {
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);

      // Each thread allocates and reuses its own three compact tiles.
      std::vector<double> At(tile_size), Bt(tile_size), Ct(tile_size);

      // Each output tile has one owner, which accumulates every K contribution.
#pragma omp for collapse(2)
      for (int I = 0; I < n; I += b) {
         for (int J = 0; J < n; J += b) {
            // Accumulate all K contributions to C tile.
            for (int i = 0; i < b; ++i) {
               for (int j = 0; j < b; ++j) {
                  Ct[i * b + j] = C[(I + i) * n + (J + j)];
               }
            }

            for (int K = 0; K < n; K += b) {
               // Copy each source tile row by row: global stride n, local stride b.
               for (int i = 0; i < b; ++i) {
                  for (int j = 0; j < b; ++j) {
                     At[i * b + j] = A[(I + i) * n + (K + j)];
                     Bt[i * b + j] = B[(K + i) * n + (J + j)];
                  }
               }

               for (int i = 0; i < b; ++i) {
                  for (int j = 0; j < b; ++j) {
                     for (int k = 0; k < b; ++k) {
                        Ct[i * b + j] += At[i * b + k] * Bt[k * b + j];
                     }
                  }
               }
            }

            // Write back only after all partial products have accumulated.
            for (int i = 0; i < b; ++i) {
               for (int j = 0; j < b; ++j) {
                  C[(I + i) * n + (J + j)] = Ct[i * b + j];
               }
            }
         }
      }

      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
   }
}
