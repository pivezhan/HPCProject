


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
    B_L2 = 
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