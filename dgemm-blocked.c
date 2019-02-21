#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const char *dgemm_desc = "Simple blocked dgemm.";

#define min(a,b) (((a)<(b))?(a):(b))
#define BLOCK_SIZEReg 4
#define BLOCK_SIZEL1 80 // Should be multiple of BLOCK_SIZEReg
#define BLOCK_SIZEL2 512
// #define blocksize 10

double AL1[BLOCK_SIZEL1 *BLOCK_SIZEL1] __attribute__((aligned(16)));
double BL1[BLOCK_SIZEL1 *BLOCK_SIZEL1] __attribute__((aligned(16)));
double CL1[BLOCK_SIZEL1 *BLOCK_SIZEL1] __attribute__((aligned(16)));
// double AL2[BLOCK_SIZEL2 *BLOCK_SIZEL2] __attribute__((aligned(16)));
double BL2[BLOCK_SIZEL2 *BLOCK_SIZEL2] __attribute__((aligned(16)));

// double AREG[BLOCK_SIZEReg *BLOCK_SIZEReg] __attribute__((aligned(16)));
// double BREG[BLOCK_SIZEReg *BLOCK_SIZEReg] __attribute__((aligned(16)));

void PrintMatrix(double* a, int n, int m)
{
  for (int j = 0; j < m; j++)
  {
    for (int i = 0; i < n; i++)
    {
        printf("%lf ",a[j*n + i]);
    }
    printf("\n");
  }
}


double CreateSqMatrix(double* a, int n)
{
    int temp = 0;
    for (int i = 0; i<n; ++i)
        for (int j = 0; j<n; ++j)
        {
            a[j+i*n] = temp;
            temp += 1;
        }
    return *a;
}

void Multiply(double* a, double* b, double *c, int n)
{
    for (int i = 0; i<n; ++i)
        for (int j = 0; j<n; ++j)
            for (int k=0; k<n; ++k)
                c[j+i*n] += a[k+i*n]*b[j+k*n];
    // return *c;
}

// void MatrixBlockTest(int n, double* a, double* b, double* c)
// {
//     double B_limit = 1 + ((n - 1)/blocksize);
//     // printf("n: %d,  blocksize: %d\n", n, blocksize);
//     // printf("Blimit %lf\n", B_limit);
//     for (int ii = 0; ii<B_limit; ++ii)
//     {
//         for (int jj = 0; jj<B_limit; ++jj)
//         {
//             for (int kk = 0; kk<B_limit; ++kk)
//             {
//                 int small_blocksize = min(blocksize, n - (kk)*blocksize);
//                 // printf("-----------------------small_blocksize: %d\n", small_blocksize);
//                 for (int i = 0; i<blocksize; ++i)
//                     for (int j = 0; j<blocksize; ++j)
//                         for (int k=0; k<small_blocksize; ++k)
//                         {
//                             // printf("k %d j %d i %d kk %d jj %d ii %d\n", k, j, i, kk, jj, ii);
//                             // printf("test -------- %d+%d*%d+(%d*%d+%d*%d*2)\n", j,k,n,jj,blocksize,kk,n);
//                             c[j+i*n+(jj*blocksize+ii*n*blocksize)] += a[k+i*n+(kk*blocksize+ii*n*blocksize)]*b[j+k*n+(jj*blocksize+kk*n*blocksize)];
//                             // printf("C[%d] += A[%d]*B[%d]\n", j+i*n+(jj*blocksize+ii*n*blocksize), k+i*n+(kk*blocksize+ii*n*blocksize), j+k*n+(jj*blocksize+kk*n*blocksize));
//                             // printf("%lf += %lf * %lf\n",)
//                         }
//             }
//         }
//     }
// }

// load_l1_block(BLOCK_SIZEL2, K_, N_, BL1, B + k + j * BLOCK_SIZEL2);
static inline void load_l1_block(int K, int M, int N, double *to, double *from)
{
    // printf("-------------------LOAD_BLOCK_L1----------------------\n");
    int j = 0;
    for (; j < N; j++)
    {
        int i = M;
        // printf("B[%d]: %p\n", j * K, (from + j * K));
        memcpy(to + j * BLOCK_SIZEL1, from + j * K, sizeof(double)*M);
        memset(to + i + j * BLOCK_SIZEL1, 0, sizeof(to) * (BLOCK_SIZEL1 - i));
    }
    for (; j < BLOCK_SIZEL1; j++)
    {
        memset(to + j * BLOCK_SIZEL1, 0, sizeof(to)*BLOCK_SIZEL1);
    }
}

static inline void save_l1_block(int K, int M, int N, double *to, double *from)
{
    for (int j = 0; j < N; j++)
    {
        memcpy(to + j * K, from + j * BLOCK_SIZEL1, sizeof(double)*M);
    }
}

// load_l2_block(lda, K, N, BL2, B + k + j * lda);
static inline void load_l2_block(int K, int M, int N, double *to, double *from)
{
    for (int j = 0; j < N; j++)
    {
        memcpy(to + j * BLOCK_SIZEL2, from + j * K, sizeof(double)*M);
    }
}



static inline void do_block_l1 (int lda, int M, int N, int K, double *A, double *B, double *C)
{
    // printf("-------------------DO_BLOCK_L1----------------------\n");
    /* For each row i of A */
    for (int j = 0; j < N; j += BLOCK_SIZEReg)
    {
        /* For each column j of B */
        for (int k = 0; k < K; k += BLOCK_SIZEReg)
        {
            // =========================================
            //FOR BLOCK_SIZEReg == 2
            // =========================================

#if BLOCK_SIZEReg == 2
            // printf("j: %d, k: %d, j * BLOCK_SIZEL1 + k: %d\n", j, k, j * BLOCK_SIZEL1 + k);
            double *restrict BB = B + j * BLOCK_SIZEL1 + k;
            register double b00 = BB[0];
            register double b10 = BB[1];
            register double b01 = BB[BLOCK_SIZEL1];
            register double b11 = BB[1 + BLOCK_SIZEL1];
            // printf("B[0]: %p, %lf\n", &(B[0]), (B[0]));
            // printf("B[1]: %p, %lf\n", &(B[1]), (B[1]));
            // printf("B[%d]: %p, %lf\n", BLOCK_SIZEL1, &(B[BLOCK_SIZEL1]), (B[BLOCK_SIZEL1]));
            // printf("B[%d]: %p, %lf\n", 1 + BLOCK_SIZEL1, &(B[1 + BLOCK_SIZEL1]), (B[1 + BLOCK_SIZEL1]));
            double *restrict AA = A + k * BLOCK_SIZEL1;
            double *restrict CC = C + j * BLOCK_SIZEL1;
            // printf("A[%d]: %p, %lf\n", k * BLOCK_SIZEL1, &(A[k * BLOCK_SIZEL1]), (A[k * BLOCK_SIZEL1]));
            // printf("C[%d]: %p, %lf\n", j * BLOCK_SIZEL1, &(C[j * BLOCK_SIZEL1]), (C[j * BLOCK_SIZEL1]));
            int i = 0;

            for (i = 0; i < BLOCK_SIZEL1; i++)
            {
                // printf("-----------------------\n");
                // printf("A[%d]: %p, %lf\n", i, &(A[i]), (*(AA + i)));
                // printf("A[%d]: %p, %lf\n", i + BLOCK_SIZEL1, &(A[i + BLOCK_SIZEL1]), (*(AA + i + BLOCK_SIZEL1)));
                register double a0 = *(AA + i);
                register double a1 = *(AA + i + BLOCK_SIZEL1);
                *(CC + i) += b00 * a0 + b10 * a1;
                *(CC + i + BLOCK_SIZEL1) += b01 * a0 + b11 * a1;
                // printf("C[%d] += %lf * %lf + %lf * %lf\n", i, b00, a0, b10, a1);
                // printf("C[%d] += %p * %p + %p * %p\n", i, &b00 * &a0 + &b10 * &a1);
                // printf("C[%d]: %p, %lf\n", i, &(C[i]), (*(CC + i)));
                // printf("C[%d]: %p, %lf\n", i + BLOCK_SIZEL1, &(C[i + BLOCK_SIZEL1]), (*(CC + i + BLOCK_SIZEL1)));
                // printf("-----------------------\n");
            }
#endif
            // =========================================
            //FOR BLOCK_SIZEReg == 4
            // =========================================
#if BLOCK_SIZEReg == 4


            double *restrict BB = B + j * BLOCK_SIZEL1 + k;
            double *restrict AA = A + k * BLOCK_SIZEL1;
            double *restrict CC = C + j * BLOCK_SIZEL1;

            register double b00 = BB[0];
            register double b10 = BB[1];
            register double b20 = BB[2];
            register double b30 = BB[3];
            register double b01 = BB[0 + BLOCK_SIZEL1];
            register double b11 = BB[1 + BLOCK_SIZEL1];
            register double b21 = BB[2 + BLOCK_SIZEL1];
            register double b31 = BB[3 + BLOCK_SIZEL1];
            register double b02 = BB[0 + 2 * BLOCK_SIZEL1];
            register double b12 = BB[1 + 2 * BLOCK_SIZEL1];
            register double b22 = BB[2 + 2 * BLOCK_SIZEL1];
            register double b32 = BB[3 + 2 * BLOCK_SIZEL1];
            register double b03 = BB[0 + 3 * BLOCK_SIZEL1];
            register double b13 = BB[1 + 3 * BLOCK_SIZEL1];
            register double b23 = BB[2 + 3 * BLOCK_SIZEL1];
            register double b33 = BB[3 + 3 * BLOCK_SIZEL1];

            int i = 0;

            for (i = 0; i < BLOCK_SIZEL1; i++)
            {
                register double ai0 = *(AA + i);
                register double ai1 = *(AA + i + BLOCK_SIZEL1);
                register double ai2 = *(AA + i + 2 * BLOCK_SIZEL1);
                register double ai3 = *(AA + i + 3 * BLOCK_SIZEL1);

                *(CC + i) += ai0 * b00 + ai1 * b10 + ai2 * b20 + ai3 * b30;
                *(CC + i + BLOCK_SIZEL1) += ai0 * b01 + ai1 * b11 + ai2 * b21 + ai3 * b31;
                *(CC + i + 2 * BLOCK_SIZEL1) += ai0 * b02 + ai1 * b12 + ai2 * b22 + ai3 * b32;
                *(CC + i + 3 * BLOCK_SIZEL1) += ai0 * b03 + ai1 * b13 + ai2 * b23 + ai3 * b33;
            }
#endif

        } //For k
    }//For j
}



// do_block_l2(lda, M, N, K, A + i + k * lda, BL2, C + i + j * lda);

static inline void do_block_l2 (int lda, int M, int N, int K, double *A, double *B, double *C)
{
    // printf("-------------------DO_BLOCK_L2----------------------\n");
    // #pragma omp parallel
{
    // #pragma omp for
    for (int j = 0; j < N; j += BLOCK_SIZEL1)
    {
        int N_ = min (BLOCK_SIZEL1, N - j);
        // printf("N_: %d, N: %d, BLOCK_SIZEL2: %d, lda: %d, j: %d, lda-j: %d\n", N_, N, BLOCK_SIZEL2, lda, j, lda-j);
        for (int k = 0; k < K; k += BLOCK_SIZEL1)
        {
            int K_ = min (BLOCK_SIZEL1, K - k);
            // printf("K_: %d, K: %d, BLOCK_SIZEL2: %d, lda: %d, k: %d, lda-k: %d\n", K_, K, BLOCK_SIZEL2, lda, k, lda-k);
            load_l1_block(BLOCK_SIZEL2, K_, N_, BL1, B + k + j * BLOCK_SIZEL2);
            for (int i = 0; i < M; i += BLOCK_SIZEL1)
            {
                int M_ = min (BLOCK_SIZEL1, M - i);
                // printf("M_: %d, M: %d, BLOCK_SIZEL2: %d, lda: %d, i: %d, lda-i: %d\n", M_, M, BLOCK_SIZEL2, lda, i, lda-i);
                load_l1_block(lda, M_, K_, AL1, A + i + k * lda);
                load_l1_block(lda, M_, N_, CL1, C + i + j * lda);
                do_block_l1(lda, M_, N_, K_, AL1, BL1, CL1);
                save_l1_block(lda, M_, N_, C + i + j * lda, CL1);
            }
        }
    }
}
}



void square_dgemm (int lda, double *A, double *B, double *C)
{
    // printf("-------------------SQUARE_DGEMM----------------------\n");
    for (int j = 0; j < lda; j += BLOCK_SIZEL2)
    {
        int N = min (BLOCK_SIZEL2, lda - j);
        // printf("N: %d, BLOCK_SIZEL2: %d, lda: %d, j: %d, lda-j: %d\n", N, BLOCK_SIZEL2, lda, j, lda-j);
        for (int k = 0; k < lda; k += BLOCK_SIZEL2)
        {
            int K = min (BLOCK_SIZEL2, lda - k);
            // printf("K: %d, BLOCK_SIZEL2: %d, lda: %d, k: %d, lda-k: %d\n", K, BLOCK_SIZEL2, lda, k, lda-k);
            load_l2_block(lda, K, N, BL2, B + k + j * lda);
            for (int i = 0; i < lda; i += BLOCK_SIZEL2)
            {
                int M = min (BLOCK_SIZEL2, lda - i);
                // printf("M: %d, BLOCK_SIZEL2: %d, lda: %d, i: %d, lda-i: %d\n", M, BLOCK_SIZEL2, lda, i, lda-i);
                do_block_l2(lda, M, N, K, A + i + k * lda, BL2, C + i + j * lda);
            }
        }
    }

}


// void main()
// {
//     int nmax = 769;
//     int n = 769;
//     double *buf = NULL;
//     buf = (double*) malloc (3*nmax*nmax*(sizeof(double)));

//     double *A;
//     double *B;
//     double *C;
//     A = buf;
//     B = A + nmax*nmax;
//     C = B + nmax*nmax;
    
//     // fill (A, n*n);
//     // fill (B, n*n);
//     // fill (C, n*n);
//     *A = CreateSqMatrix(A, n);
//     // printf("A \n");
//     // PrintMatrix(A, n, n);

//     *B = CreateSqMatrix(B, n);
//     // printf("B \n");
//     // PrintMatrix(B, n, n);

  

//     clock_t t; 
//     t = clock(); 
//     square_dgemm(n,A,B,C);
//     // Multiply(A,B,C,n);
//     t = clock() - t; 
    
//     // printf("C \n");
//     // PrintMatrix(C, n, n);

//     double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds 
//     printf("Took %f seconds to execute \n", time_taken);
// }