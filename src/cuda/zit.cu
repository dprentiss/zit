#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

static const uint NUM_BUYERS = 1 << 10;
static const uint MAX_BUYER_VALUE = 20;
static const uint MAX_SELLER_VALUE = MAX_BUYER_VALUE;

//static const int MAX_TRANSACTIONS = 1E5;

// declare pointers to int arrays
int *buyerValues;
int *sellerValues;
int *transactionPrice;

__global__
void trade(int *buyerValues, int *sellerValues, int *transactionPrice)
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
