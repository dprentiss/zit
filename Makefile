NVCC = nvcc
SRC_DIR = src
BIN_DIR = bin
TARGET = zit

all: build

build: 
	$(NVCC) $(SRC_DIR)/*.cu -o $(BIN_DIR)/$(TARGET)

clean: 
	rm -r $(BIN_DIR)
