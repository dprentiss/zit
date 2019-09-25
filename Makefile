SHELL = /bin/sh

CC = gcc
NVCC = nvcc
SRC = src
BIN = bin

.PHONY: all
all: pthreads pthreadclone zitcuda

.PHONY: pthreads
pthreads:
	mkdir -p $(BIN)/rax
	$(CC) $(SRC)/rax/zitpthreads.c -o $(BIN)/rax/zitp -lm -lpthread

.PHONY: pthreadclone
pthreadclone:
	$(RM) $(BIN)/$@
	mkdir -p $(BIN)
	$(NVCC) $(SRC)/cuda/pthreadclone/pthreadclone.cu -o $(BIN)/$@

.PHONY: zitcuda
zitcuda:
	$(RM) $(BIN)/$@
	mkdir -p $(BIN)
	$(NVCC) $(SRC)/cuda/zit.cu -o $(BIN)/$@
