#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
 
static const int NUM_BUYERS = 1E3; 
static const int MAX_BUYER_VALUE = 20; 
static const int MAX_SELLER_VALUE = 20; 
 
__global__ 
void trade(int *buyerValues, int *sellerValues, int *transactionPrice) 
{
    curandState_t state;
    curand_init(0, 0, 0, &state);

    int bid, ask;
    int idx = threadIdx.x;
    if (transactionPrice[idx] == 0) {
       bid = curand(&state) % buyerValues[idx] + 1;
       ask = sellerValues[idx] + curand(&state) % (MAX_SELLER_VALUE - sellerValues[idx] + 1);
       if (bid >= ask) {
           transactionPrice[threadIdx.x] = ask + curand(&state) % (bid - ask + 1);
       }
    }
}
 
int main()
{
    // seed random number generator
    srand(time(NULL));

    int *buyerValues;
    int *sellerValues;
    int *transactionPrice;

    // size of each array
    size_t intSize = NUM_BUYERS*sizeof(int);
    // allocate memeory on device
    cudaMallocManaged(&buyerValues, intSize);
    cudaMallocManaged(&sellerValues, intSize);
    cudaMallocManaged(&transactionPrice, intSize);


    
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

    /*
       char a[N] = "Hello \0\0\0\0\0\0";
       int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

       char *ad;
       int *bd;
       const int csize = N*sizeof(char);
       const int isize = N*sizeof(int);

       printf("%s", a);

       cudaMalloc( (void**)&ad, csize ); 
       cudaMalloc( (void**)&bd, isize ); 
       cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
       cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 

       dim3 dimBlock( blocksize, 1 );
       dim3 dimGrid( 1, 1 );
       hello<<<dimGrid, dimBlock>>>(ad, bd);
       cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
       cudaFree( ad );
       cudaFree( bd );
     */
}
