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

// void multiply(int n, int mat1[], int mat2[], int res[]]) 
// { 
//     int i, j, k; 
//     for (i = 0; i < n; i++) 
//     { 
//         for (j = 0; j < n; j++) 
//         { 
//             res[i][j] = 0; 
//             for (k = 0; k < n; k++) 
//                 res[i][j] += mat1[i][k]*mat2[k][j]; 
//         } 
//     } 
// } 

#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
const char* dgemm_desc = "Simple blocked dgemm.";
#include <immintrin.h>

#define turn_even(M) (((M)%2)?((M)+1):(M))
#define min(a,b) (((a)<(b))?(a):(b))

void fill (double* p, int n);
void naive_dgemm (int lda, double* A, double* B, double* C);

#define min(a,b) (((a)<(b))?(a):(b))
#define BLOCK_SIZE_INTERNAL 2
#define BLOCK_SIZE_L1 64
#define BLOCK_SIZE_L2 256

static inline void load_L1_Block(int K, int M, int N, double *to, double *from);
static inline void load_L2_Block(int lda, int K, int N, double *to, double *from);
static inline void save_L1_Block(int K, int M, int N, double *to, double *from);
static inline void do_L1_Block (int lda, int M, int N, int K, double *A, double *B, double *C);
static inline void do_L2_Block (int lda, int M, int N, int K, double *A, double *B, double *C);

static inline void load_L1_Block(int K, int M, int N, double *to, double *from)
{
    for (int j = 0; j < N; j++)
    {
        int i = M;
        memcpy(to + j * BLOCK_SIZE_L1, from + j * K, sizeof(double)*M);
        memset(to + i + j * BLOCK_SIZE_L1, 0, sizeof(to) * (BLOCK_SIZE_L1 - i));
    }
    for (int j = 0; j < BLOCK_SIZE_L1; j++)
        memset(to + j * BLOCK_SIZE_L1, 0, sizeof(to)*BLOCK_SIZE_L1);
}

static inline void load_L2_Block(int lda, int K, int N, double *to, double *from)
{
    for (int j = 0; j < N; j++)
        memcpy(to + j * BLOCK_SIZE_L2, from + j * lda, sizeof(double)*K);
}

static inline void save_L1_Block(int K, int M, int N, double *to, double *from)
{
    for (int j = 0; j < N; j++)
        memcpy(to + j * K, from + j * BLOCK_SIZE_L1, sizeof(double)*M);
}

static inline void do_L1_Block (int lda, int M, int N, int K, double *A, double *B, double *C)
{
    for (int j = 0; j < N; j += BLOCK_SIZE_INTERNAL)
    {
        for (int k = 0; k < K; k += BLOCK_SIZE_INTERNAL)
        {

            if (BLOCK_SIZE_INTERNAL == 2)
            {
                double *restrict BB = B + j * BLOCK_SIZE_L1 + k;

                register double b00 = BB[0];
                register double b01 = BB[1];
                register double b10 = BB[BLOCK_SIZE_L1];
                register double b11 = BB[1 + BLOCK_SIZE_L1];

                double *restrict AA = A + k * BLOCK_SIZE_L1;
                double *restrict CC = C + j * BLOCK_SIZE_L1;

                int i = 0;

                for (i = 0; i < BLOCK_SIZE_L1; i++)
                {
                    register double a0 = *(AA + i);
                    register double a1 = *(AA + i + BLOCK_SIZE_L1);
                    *(CC + i) += b00 * a0 + b01 * a1;
                    *(CC + i + BLOCK_SIZE_L1) += b10 * a0 + b11 * a1;
                }
            }

            else if (BLOCK_SIZE_INTERNAL == 4)
            {
                double *restrict BB = B + j * BLOCK_SIZE_L1 + k;
                double *restrict AA = A + k * BLOCK_SIZE_L1;
                double *restrict CC = C + j * BLOCK_SIZE_L1;

                register double b00 = BB[0];
                register double b01 = BB[1];
                register double b02 = BB[2];
                register double b03 = BB[3];

                register double b10 = BB[0 + BLOCK_SIZE_L1];
                register double b11 = BB[1 + BLOCK_SIZE_L1];
                register double b12 = BB[2 + BLOCK_SIZE_L1];
                register double b13 = BB[3 + BLOCK_SIZE_L1];

                register double b20 = BB[0 + 2 * BLOCK_SIZE_L1];
                register double b21 = BB[1 + 2 * BLOCK_SIZE_L1];
                register double b22 = BB[2 + 2 * BLOCK_SIZE_L1];
                register double b23 = BB[3 + 2 * BLOCK_SIZE_L1];

                register double b30 = BB[0 + 3 * BLOCK_SIZE_L1];
                register double b31 = BB[1 + 3 * BLOCK_SIZE_L1];
                register double b32 = BB[2 + 3 * BLOCK_SIZE_L1];
                register double b33 = BB[3 + 3 * BLOCK_SIZE_L1];

                int i = 0;

                for (i = 0; i < BLOCK_SIZE_L1; i++)
                {
                    register double ai0 = *(AA + i);
                    register double ai1 = *(AA + i + BLOCK_SIZE_L1);
                    register double ai2 = *(AA + i + 2 * BLOCK_SIZE_L1);
                    register double ai3 = *(AA + i + 3 * BLOCK_SIZE_L1);

                    *(CC + i) += ai0 * b00 + ai1 * b01 + ai2 * b02 + ai3 * b03;
                    *(CC + i + BLOCK_SIZE_L1) += ai0 * b10 + ai1 * b11 + ai2 * b12 + ai3 * b13;
                    *(CC + i + 2 * BLOCK_SIZE_L1) += ai0 * b20 + ai1 * b21 + ai2 * b22 + ai3 * b23;
                    *(CC + i + 3 * BLOCK_SIZE_L1) += ai0 * b30 + ai1 * b31 + ai2 * b32 + ai3 * b33;
                }
            }
        }
    }
}

static inline void do_L2_Block (int lda, int M, int N, int K, double *A, double *B, double *C)
{
double B_L1[BLOCK_SIZE_L1*BLOCK_SIZE_L1];
double A_L1[BLOCK_SIZE_L1*BLOCK_SIZE_L1];
double C_L1[BLOCK_SIZE_L1*BLOCK_SIZE_L1];
    for (int j = 0; j < N; j += BLOCK_SIZE_L1)
    {
        int N_ = min (BLOCK_SIZE_L1, N - j);
        for (int k = 0; k < K; k += BLOCK_SIZE_L1)
        {
            int K_ = min (BLOCK_SIZE_L1, K - k);
            load_L1_Block(BLOCK_SIZE_L2, K_, N_, B_L1, B + k + j * BLOCK_SIZE_L2);
            for (int i = 0; i < M; i += BLOCK_SIZE_L1)
            {
                int M_ = min (BLOCK_SIZE_L1, M - i);
                load_L1_Block(lda, M_, K_, A_L1, A + i + k * lda);
                load_L1_Block(lda, M_, N_, C_L1, C + i + j * lda);
                do_L1_Block(lda, M_, N_, K_, A_L1, B_L1, C_L1);
                save_L1_Block(lda, M_, N_, C + i + j * lda, C_L1);
            }
        }
    }
}



void square_dgemm (int lda, double *A, double *B, double *C)
{
    double B_L2[BLOCK_SIZE_L2*BLOCK_SIZE_L2];
    for (int j = 0; j < lda; j += BLOCK_SIZE_L2)
    {
        int N = min (BLOCK_SIZE_L2, lda - j);
        for (int k = 0; k < lda; k += BLOCK_SIZE_L2)
        {
            int K = min (BLOCK_SIZE_L2, lda - k);
            load_L2_Block(lda, K, N, B_L2, B + k + j * lda);
            for (int i = 0; i < lda; i += BLOCK_SIZE_L2)
            {
                int M = min (BLOCK_SIZE_L2, lda - i);
                do_L2_Block(lda, M, N, K, A + i + k * lda, B_L2, C + i + j * lda);
            }
        }
    }
}



//////////////// Testing the code ///////////////////
int main() 
{ 
int lda=5;
    double mat1[] = { 1.0, .5, 2, 1, 1, 2, 2, 1, 2, 1, 3, 3, 3, 3, 1, 4, 4, 4, 4, 1, 4, 0, 4, 1, 1};
  
    double mat2[] = {1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 1, 4, 4, 4, 4, 2, 4, 1, 4, 2, 1}; 
  
    double res[lda*lda]; // To store result 
    int i, j; 
//(int) sizeof(mat1)

    square_dgemm(lda, &mat1[0], &mat2[0], &res[0]); 
    printf("First input is: \n"); 
    for (i = 0; i < lda; i++) 
    { 
        for (j = 0; j < lda; j++) 
           printf("%f ", mat1[i+j*lda]); 
        printf("\n"); 
    } 
      printf("Second input is: \n"); 
    for (i = 0; i < lda; i++) 
    { 
        for (j = 0; j < lda; j++) 
           printf("%f ", mat2[i+j*lda]); 
        printf("\n"); 
    } 
    printf("Result matrix is \n"); 
    for (i = 0; i < lda; i++) 
    { 
        for (j = 0; j < lda; j++) 
           printf("%f ", res[i+j*lda]); 
        printf("\n"); 
    } 
// Speed Test
int test_sizes[] = 

  /* Multiples-of-32, +/- 1. Currently commented. */
  /* {31,32,33,63,64,65,95,96,97,127,128,129,159,160,161,191,192,193,223,224,225,255,256,257,287,288,289,319,320,321,351,352,353,383,384,385,415,416,417,447,448,449,479,480,481,511,512,513,543,544,545,575,576,577,607,608,609,639,640,641,671,672,673,703,704,705,735,736,737,767,768,769,799,800,801,831,832,833,863,864,865,895,896,897,927,928,929,959,960,961,991,992,993,1023,1024,1025}; */

  /* A representative subset of the first list. Currently uncommented. */ 
  { 31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
    319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769 };

  int nsizes = sizeof(test_sizes)/sizeof(test_sizes[0]);

  /* assume last size is also the largest size */
  int nmax = test_sizes[nsizes-1];

  /* allocate memory for all problems */
  double* buf = NULL;
  buf = (double*) malloc (3 * nmax * nmax * sizeof(double));


  /* For each test size */
    /* Create and fill 3 random matrices A,B,C*/
    int n = test_sizes[20];

    double* A = buf + 0;
    double* B = A + nmax*nmax;
    double* C = B + nmax*nmax;

    fill (A, n*n);
    fill (B, n*n);
    fill (C, n*n);

double time;
double start = omp_get_wtime();
    naive_dgemm(n, &A[0], &B[0], &C[0]); 
    double stop = omp_get_wtime();
time=stop-start;
printf("The elapsed time is %f sec\n", time); 


    return 0; 
}

void fill (double* p, int n)
{
  for (int i = 0; i < n; ++i)
    p[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
}

void naive_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < lda; ++i)
    /* For each column j of B */
    for (int j = 0; j < lda; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for( int k = 0; k < lda; k++ )
  cij += A[i+k*lda] * B[k+j*lda];
      C[i+j*lda] = cij;
    }
}
