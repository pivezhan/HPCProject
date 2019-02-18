/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

const char* dgemm_desc = "Simple blocked dgemm.";
#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#define BLOCK_L1 100
#define BLOCK_L2 290
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
//function declaration
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C);
void square_dgemm (int lda, double* A, double* B, double* C);
static void transposeA(double* restrict AT, double* restrict A, int lda, int M, int K);
static void do_block_unrollavx128(const int lda, const int M, const int N, const int K, double* restrict A, double* restrict B, double* restrict C);


static void transposeA(double* restrict AT, double* restrict A, int lda, int M, int K)
{
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      AT[k + i*lda] = A[i + k*lda];
    }   
  }   
}

static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
//double AT[M*K];
//transposeA(AT, A, lda, M, K);
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
	cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}



/* UNROLL 2: This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_unrollavx128(int lda, int M, int N, int K, double* A, double* B, double* C)
{
 __m128d c0, c1, a0, a1, b0, b1, b2, b3, d0, d1; 

  for (int k = 0; k < K; k += 2 ) {
    for (int j = 0; j < N; j += 2) {
      double Bkj = _mm_load1_pd(k+j*lda);
      double Bkj_1 = _mm_load1_pd(k+1+j*lda);
      double Bkj_2 = _mm_load1_pd(k+(j+1)*lda);
      double Bkj_3 = _mm_load1_pd(k+1+(j+1)*lda);
      for (int i = 0; i < M; i += 2) {
        Aik = _mm_load_pd(A+i+k*lda);
        Aik_1 = _mm_load_pd(A+i+(k+1)*lda);

        Cij = _mm_load_pd(C+i+j*lda);
        Cij_1 = _mm_load_pd(C+i+(j+1)*lda);

        d0 = _mm_add_pd(Cij, _mm_mul_pd(Aik,Bkj));
        d1 = _mm_add_pd(Cij_1, _mm_mul_pd(Aik,Bkj_2));
        Cij = _mm_add_pd(d0, _mm_mul_pd(Aik_1,Bkj_1));
        Cij_1 = _mm_add_pd(d1, _mm_mul_pd(Aik_1,Bkj_3));
        _mm_store_pd(C+i+j*lda,Cij);
        _mm_store_pd(C+i+(j+1)*lda,Cij_1);
      }
    }
  }
}



/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  

void square_dgemm (int lda, double* A, double* B, double* C)
{
	/* For each L2-sized block-row of A */ 
	for (int r = 0; r < lda; r += BLOCK_L2) 
	{
	int end_i = r + min(BLOCK_L2, lda-r);
		/* For each L2-sized block-column of B */
		for (int s = 0; s < lda; s += BLOCK_L2) 
		{
		int end_j = s + min(BLOCK_L2, lda-s);
			/* Accumulate L2-sized block dgemms into block of C */
			for (int t = 0; t < lda; t += BLOCK_L2) 
			{
			int end_k = t + min(BLOCK_L2, lda-t);
				/* For each L1-sized block-row of A */ 
				for (int i = r; i < end_i; i += BLOCK_L1) 
				{
					/* For each L1-sized block-column of B */
					for (int j = s; j < end_j; j += BLOCK_L1) 
					{
						/* Accumulate L1-sized block dgemms into block of C */
						for (int k = t; k < end_k; k += BLOCK_L1) 
						{
						int K = min(BLOCK_L1, end_k-k);
						int N = min(BLOCK_L1, end_j-j);
						int M = min(BLOCK_L1, end_i-i);
						/* Performs a smaller dgemm operation
						*  C' := C' + A' * B'
						* where C' is M-by-N, A' is M-by-K, and B' is K-by-N. */
						do_block_unrollavx128(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
						}
					}
				}
			}
		}
	}
}

