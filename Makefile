CC = cc
NVCC = nvcc
SRC_DIR = src
BIN_DIR = bin
TARGET = zit

all: clean build

build: 
	$(NVCC) $(SRC_DIR)/*.cu -o $(BIN_DIR)/$(TARGET)

clean: 
	$(RM) -r $(BIN_DIR)/*

mkdir: 
	mkdir -p $(BIN_DIR)

pthreads: 
	mkdir -p $(BIN_DIR)
	mkdir -p $(BIN_DIR)/rax
	$(CC) $(SRC_DIR)/rax/zitpthreads.c -o $(BIN_DIR)/rax/zitp -lm -lpthread

run:
	./$(BIN_DIR)/$(TARGET)
