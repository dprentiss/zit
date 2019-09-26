#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

static const unsigned int NUM_BUYERS = 1 << 10;
static const unsigned int MAX_BUYER_VALUE = 20;
static const unsigned int MAX_SELLER_VALUE = MAX_BUYER_VALUE;

// static const unsigned int MAX_TRADES = 1 << 10;

unsigned int *buyerValues;
unsigned int *sellerValues;
unsigned int *transactionPrice;
curandState *states;

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
void key_split(unsigned int *a, unsigned int bit, unsigned int *key)
{
    unsigned int idx = threadIdx.x;
    unsigned int N = blockDim.x;
    unsigned int key_idx = key[idx];
    unsigned int a_idx = a[idx];
    unsigned int b_idx = (key_idx >> bit) & 1;

    key[idx] = b_idx;

    __syncthreads();

    unsigned int T_before = scan(key);
    unsigned int T_total = key[N-1];
    unsigned int F_total = N - T_total;

    __syncthreads();

    if (b_idx) {
        key[T_before-1+F_total] = key_idx;
        a[T_before-1+F_total] = a_idx;
    } else {
        key[idx-T_before] = key_idx;
        a[idx-T_before] = a_idx;
    }
}

    __device__
void key_sort(unsigned int *a, unsigned int *key)
{
    unsigned int bit;
    size_t n = CHAR_BIT * sizeof(a[0]);

    for (bit = 0; bit < n; ++bit) {
        key_split(a, bit, key);
        __syncthreads();
    }
}

    __global__
void init(unsigned int *buyerValues,
        unsigned int *sellerValues,
        unsigned int *transactionPrice,
        curandState *states,
        unsigned long seed)
{
    int idx = threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
    curandState state = states[idx];

    // initialize buyers and sellers with random value
    // initialize transactionPrice with 0
    // random value for buyers
    buyerValues[idx] = curand(&state) % MAX_BUYER_VALUE + 1;
    // random value for traders
    sellerValues[idx] = curand(&state) % MAX_SELLER_VALUE + 1;
    // zero indicates no trade has taken place between buyer and seller i
    transactionPrice[idx] = 0;
}

__global__
//__device__
void shuffle(unsigned int *buyerValues,
        unsigned int *sellerValues,
        unsigned int *transactionPrice,
        curandState *states)
{
    unsigned int b_tmp;
    unsigned int s_tmp;
    unsigned int p_tmp;
    __shared__ unsigned int key[NUM_BUYERS];
    __shared__ unsigned int perm[NUM_BUYERS];
    int idx = threadIdx.x;
    curandState state = states[idx];
    perm[idx] = idx;


    if (transactionPrice[idx] == 0) {
        key[idx] = idx + NUM_BUYERS;
    } else {
        key[idx] = idx;
    }

    __syncthreads();

    key_sort(perm, key);

    __syncthreads();

    b_tmp = buyerValues[perm[idx]];
    s_tmp = sellerValues[perm[idx]];
    p_tmp = transactionPrice[perm[idx]];

    __syncthreads();    

    sellerValues[idx] = s_tmp;
    buyerValues[idx] = b_tmp;
    transactionPrice[idx] = p_tmp;

    perm[idx] = idx;

    if (transactionPrice[idx] == 0) {
        key[idx] = curand(&state);
    } else {
        key[idx] = idx;
    }

    __syncthreads();

    key_sort(perm, key);

    __syncthreads();

    s_tmp = sellerValues[perm[idx]];

    __syncthreads();    

    sellerValues[idx] = s_tmp;

    __syncthreads();    
}

    __global__
void trade(unsigned int *buyerValues,
        unsigned int *sellerValues,
        unsigned int *transactionPrice,
        curandState *states)
{
    int idx = threadIdx.x;
    int bid, ask;

    curandState state = states[idx];

    // TODO does this cause branch diversion?
    if (transactionPrice[idx] == 0) {
        bid = curand(&state) % buyerValues[idx] + 1;
        ask = sellerValues[idx] + curand(&state) % (MAX_SELLER_VALUE - sellerValues[idx] + 1);
        if (bid >= ask) {
            transactionPrice[idx] = ask + curand(&state) % (bid - ask + 1);
        }
    }

    __syncthreads();

    //shuffle(buyerValues, sellerValues, transactionPrice, states);
}

__global__
void stats(unsigned int *transactionPrice, unsigned int numTrades, unsigned int price) {
    int idx = threadIdx.x;
    unsigned int traded = 1;

    if (transactionPrice == 0) traded = 0;
}

int main()
{
    unsigned long int seed = 0;
    size_t uintSize = NUM_BUYERS*sizeof(unsigned int); // size of market array
    size_t stateSize = NUM_BUYERS*sizeof(curandState); // size of state array   
    //unsigned int trades = 0;
    unsigned int buyersRemaining = NUM_BUYERS; // number of buyers left

    // allocate managed memeory on device
    // TODO implement error handling
    cudaMallocManaged(&buyerValues, uintSize);
    cudaMallocManaged(&sellerValues, uintSize);
    cudaMallocManaged(&transactionPrice, uintSize);
    cudaMallocManaged(&states, stateSize);

    init<<<1, NUM_BUYERS>>>(buyerValues, sellerValues, transactionPrice, states, seed);

    cudaDeviceSynchronize();

    printf("Zero Intelligence Traders\n");

    /*
       for (int i = 0; i < NUM_BUYERS; i++) {
       printf("%3u", buyerValues[i]);
       }
       printf("\n");
       for (int i = 0; i < NUM_BUYERS; i++) {
       printf("%3u", sellerValues[i]);
       }
       printf("\n");
       printf("\n");
     */

    for (int i = 0; i < 1<<7; ++i) {
        trade<<<1, buyersRemaining>>>(buyerValues, sellerValues, transactionPrice, states);
        shuffle<<<1, buyersRemaining>>>(buyerValues, sellerValues, transactionPrice, states);
    }

    cudaDeviceSynchronize();

    

    /*
       for (int i = 0; i < NUM_BUYERS; i++) {
       printf("%3u", buyerValues[i]);
       }
       printf("\n");
       for (int i = 0; i < NUM_BUYERS; i++) {
       printf("%3u", sellerValues[i]);
       }
       printf("\n");
       for (int i = 0; i < NUM_BUYERS; i++) {
       printf("%3u", transactionPrice[i]);
       }
       printf("\n");
     */

    // free memory
    cudaFree(buyerValues);
    cudaFree(sellerValues);
    cudaFree(transactionPrice);

    return EXIT_SUCCESS;
}
