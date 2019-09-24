#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

static const uint NUM_BUYERS = 1 << 10;
static const uint MAX_BUYER_VALUE = 20;
static const uint MAX_SELLER_VALUE = MAX_BUYER_VALUE;

// static const int MAX_TRANSACTIONS = 1E5;

/*
 * Replaces the the values in the input array with a prefix sum.
 * Adapted from work by Mark Harris, NVIDIA and Stewart Weiss, CUNY.
 */

__device__
int scan(unsigned int *a)
{
    unsigned int idx = threadIdx.x;
    unsigned int n = blockDim.x;
    unsigned int d;

    for (d = 1; d < n; d *= 2) {
        int tmp;

        if (idx >= d)
            tmp = a[idx-d];

        __syncthreads();

        if (idx >= d)
            a[idx] = tmp + a[idx];

        __syncthreads();
    }

    return a[idx];
}

__device__
void split(unsigned int *a, unsigned int bit)
{
    unsigned int idx = threadIdx.x;
    unsigned int N = blockDim.x;
    unsigned int a_idx = a[idx];
    unsigned int b_idx = (a_idx >> bit) & 1;

    a[idx] = b_idx;

    __syncthreads();

    unsigned int T_before = scan(a);
    unsigned int T_total = a[N-1];
    unsigned int F_total = N - T_total;

    __syncthreads();

    if (b_idx) {
      a[T_before-1+F_total] = a_idx;
    } else {
      a[idx-T_before] = a_idx;
    }
}

__device__
void sort(unsigned int *a)
{
  unsigned int bit;
  size_t n = CHAR_BIT * sizeof(a[0]);

  for (bit = 0; bit < n; ++bit) {
    split(a, bit);
    __syncthreads();
  }
}

// declare pointers to int arrays
unsigned int *buyerValues;
unsigned int *sellerValues;
unsigned int *transactionPrice;

__global__
void trade(unsigned int *buyerValues,
           unsigned int *sellerValues,
           unsigned int *transactionPrice)
{
    curandState_t state;
    curand_init(0, 0, 0, &state);

    int idx = threadIdx.x;
    int bid, ask;

    // TODO does this cause branch diversion?
    if (transactionPrice[idx] == 0) {
       bid = curand(&state) % buyerValues[idx] + 1;
       ask = sellerValues[idx] + curand(&state) % (MAX_SELLER_VALUE - sellerValues[idx] + 1);
       if (bid >= ask) {
           transactionPrice[idx] = ask + curand(&state) % (bid - ask + 1);
       }
    }

    __syncthreads();

    sort(sellerValues);
}

int main()
{
    // seed random number generator
    srand(time(NULL));

    // size of each array
    size_t intSize = NUM_BUYERS*sizeof(int);
    // allocate managed memeory on device
    // TODO implement error handling
    cudaMallocManaged(&buyerValues, intSize);
    cudaMallocManaged(&sellerValues, intSize);
    cudaMallocManaged(&transactionPrice, intSize);

    // initialize buyers and sellers with random value
    // initialize transactionPrice with 0
    for (int i = 0; i < NUM_BUYERS; i++) {
        // random value for buyers
        buyerValues[i] = rand() % MAX_BUYER_VALUE + 1;
        // random value for traders
        sellerValues[i] = rand() % MAX_SELLER_VALUE + 1;
        // zero indicates no trade has taken place between buyer and seller i
        transactionPrice[i] = 0;
    }

    trade<<<1, NUM_BUYERS>>>(buyerValues, sellerValues, transactionPrice);

    cudaDeviceSynchronize();

    printf("Zero Intelligence Trader\n");

    for (int i = 0; i < NUM_BUYERS; i++) {
        printf("%3d", buyerValues[i]);
    }
    printf("\n");
    for (int i = 0; i < NUM_BUYERS; i++) {
        printf("%3d", sellerValues[i]);
    }
    printf("\n");
    for (int i = 0; i < NUM_BUYERS; i++) {
        printf("%3d", transactionPrice[i]);
    }
    printf("\n");

    // free memory
    cudaFree(buyerValues);
    cudaFree(sellerValues);
    cudaFree(transactionPrice);

    return EXIT_SUCCESS;
}
