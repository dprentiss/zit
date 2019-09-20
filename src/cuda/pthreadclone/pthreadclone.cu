/*
	Zero Intelligence Traders

  Implemented for CUDA devices for comparison with CPU/pthreads version.
  Intended to follow logically as closely as possbile to a CPU version written
  by Robert Axtell.

  Previous code used with permission by:

	Robert Axtell

	The Brookings Institution
		and
	George Mason University
*/

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define false 0
#define true 1

#define seedRandomWithTime false	//	if false, seed the generator with 'seed'
#define seed 1

#define buyer true
#define seller false

//	Specify the maximum internal values...
#define maxBuyerValue 20
#define maxSellerValue 20

//	Specify the number of agents of each type...
#define numberOfBuyers 1000000
#define numberOfSellers 1000000

#define MaxNumberOfTrades 200000000

#define numThreads 64

//	Define an agent...
typedef struct
{
	int buyerOrSeller;
	int quantityHeld;
	int value;
	int price;
} Agent;

//	Declare the agent populations...
/*
Agent Buyers[numberOfBuyers];
Agent Sellers[numberOfSellers];
*/
Agent *Buyers;
Agent *Sellers;


const int agentsPerThread = numberOfBuyers/numThreads;
const int tradesPerThread = MaxNumberOfTrades/numThreads;

//  Seeds for one random number generator per thread
unsigned int seeds[numThreads];

/////////////////
//
//	Procedures...
//
/////////////////

void InitializeMiscellaneous()
//
//	Initialize the random number generator; cannot use srand() or rand() as these are not thread safe
{
  unsigned int i;
  for (i = 0; i<numThreads; i++)
    if (seedRandomWithTime)
      seeds[i] = (unsigned int) time(NULL);
    else
      seeds[i] = seed + i;
}	//	InitializeMiscellaneous()

void InitializeAgents()
//
//	Fill the agent fields...
//
{
  // allocate managed memory for buyers and sellers
  size_t agentSize = numberOfBuyers * sizeof(Agent);
  cudaMallocManaged(&Buyers, agentSize);
  cudaMallocManaged(&Sellers, agentSize);

	int i;

 	//	First the buyers...
 	for (i=0; i<numberOfBuyers; i=i+1)
 	{
		Buyers[i].buyerOrSeller = buyer;
		Buyers[i].quantityHeld = 0;
		Buyers[i].value = (rand_r(&seeds[0]) % maxBuyerValue) + 1;
 	};

	//	Now the sellers...
 	for (i=0; i<numberOfSellers; i=i+1)
 	{
 		Sellers[i].buyerOrSeller = seller;
 		Sellers[i].quantityHeld = 1;
 		Sellers[i].value = (rand_r(&seeds[0]) % maxSellerValue) + 1;
 	};
}	//	InitializeAgents()

__global__
void doTrades() {
//
//	This function pairs agents at random and then selects a price randomly...
//

  /*
	int i, buyerIndex, sellerIndex;
	int bidPrice, askPrice, transactionPrice;
	int threadNum = *(int*) threadN;

	int lowerBuyerBound, upperBuyerBound, lowerSellerBound, upperSellerBound;

	if (numThreads <= 10)
    printf("Thread %i up and running\n", threadNum);

	lowerBuyerBound = threadNum * agentsPerThread;
	upperBuyerBound = (threadNum + 1) * agentsPerThread - 1;
	lowerSellerBound = threadNum * agentsPerThread;
	upperSellerBound = (threadNum + 1) * agentsPerThread - 1;
	for (i=1; i<=tradesPerThread; i++)
	{
	 	//	Pick a buyer at random who has not already bought a unit,
    //  then pick a 'bid' price randomly between 1 and the agent's private value;
    //
    do {
      buyerIndex = lowerBuyerBound + rand_r(&seeds[threadNum]) % (upperBuyerBound - lowerBuyerBound);
    }
    while (Buyers[buyerIndex].quantityHeld == 1);
	 	bidPrice = (rand_r(&seeds[threadNum]) % Buyers[buyerIndex].value) + 1;

	 	//	Pick a seller at random who has not already sold a unit,
    //  then pick an 'ask' price randomly between the agent's private value and maxSellerValue;
    //
    do {
      sellerIndex = lowerSellerBound + rand_r(&seeds[threadNum]) % (upperSellerBound - lowerSellerBound);
    }
    while (Sellers[sellerIndex].quantityHeld != 1);
	 	askPrice = Sellers[sellerIndex].value + (rand_r(&seeds[threadNum]) % (maxSellerValue - Sellers[sellerIndex].value + 1));

	 	//	Let's see if a deal can be made...
	 	//
	 	if (bidPrice >= askPrice)
	 	{
	 		//	First, compute the transaction price...
      //
	 		transactionPrice = askPrice + rand_r(&seeds[threadNum]) % (bidPrice - askPrice + 1);
	 		Buyers[buyerIndex].price = transactionPrice;
	 		Sellers[sellerIndex].price = transactionPrice;
	 		//
	 		//	Then execute the exchange...
      //
	 		Buyers[buyerIndex].quantityHeld = 1;
	 		Sellers[sellerIndex].quantityHeld = 0;
	 	};
	};
  */
}

void ComputeStatistics(clock_t elapsedTime)
//
//	Determine the total quantities bought and sold...
//	...as well as statistics about prices
//
{
	int i;
	int numberBought = 0;
	int numberSold= 0;
	int sum = 0;
	double sum2 = 0.0;
	int N = 0;
	double avgPrice, sd;

	//	First, compute the quantity purchased...
  //
	for (i=0; i<numberOfBuyers; i++)
		if (Buyers[i].quantityHeld == 1)
			numberBought++;

	//	Next, get the quantity sold...
  //
	for (i=0; i<numberOfSellers; i++)
		if (Sellers[i].quantityHeld == 0)
			numberSold++;

	//	Now let's compute the average price paid as well as the standard deviation...
  //
	for (i=0; i<numberOfBuyers; i++)
		if (Buyers[i].quantityHeld == 1)
		{
			sum += Buyers[i].price;
			sum2 += pow(Buyers[i].price, 2);
			N++;
		};
	for (i=0; i<numberOfSellers; i++)
		if (Sellers[i].quantityHeld == 0)
		{
			sum += Sellers[i].price;
			sum2 += pow(Sellers[i].price, 2);
			N++;
		};
	avgPrice = (double) sum / (double) N;
	sd = sqrt((sum2 - (double) N * pow(avgPrice, 2)) / (double) (N - 1));
	printf("%i items bought and %i items sold\n", numberBought, numberSold);
	printf("The average price = %f and the s.d. is %f\n", avgPrice, sd);
	printf("The total time on CPUs was %f seconds\n", (double) elapsedTime/CLOCKS_PER_SEC);
}	//	ComputeStatistics()


void OpenMarket()
{
	clock_t startTime1, endTime1;
	time_t startTime2, endTime2;

	int threadNumber, status;
	pthread_t threads[numThreads];
	int args[numThreads];
	void *threadResult[numThreads];

	startTime1 = clock();
	time(&startTime2);

  /*
	for (threadNumber = 0; threadNumber < numThreads; threadNumber++)
	{
		args[threadNumber] = threadNumber;
		status = pthread_create(&threads[threadNumber], NULL, DoTrades, &args[threadNumber]);
		if (status != 0)
			printf("Problem launching thread %i", threadNumber);
	};

	for (threadNumber = 0; threadNumber < numThreads; threadNumber++) {
		status = pthread_join(threads[threadNumber], &threadResult[threadNumber]);
		if (status != 0)
			printf("Problem joining thread %i",threadNumber);
	};

	for (threadNumber = 0; threadNumber < numThreads; threadNumber++)
		if (threadResult[threadNumber] != 0)
			printf("Problem with termination of thread %i\n", threadNumber);
  */

	endTime1 = clock();
	time(&endTime2);

	ComputeStatistics(endTime1 - startTime1);
	endTime2 = (endTime2 - startTime2);
	printf("Wall time: %d seconds\n", (int) endTime2);

}

int main() {
  printf("\nZERO INTELLIGENCE TRADERS\n");
//  printf("%d",sizeof(Agent));

  InitializeMiscellaneous();
  InitializeAgents();

	OpenMarket();

	return(0);
}
