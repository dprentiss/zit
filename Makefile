NVCC = nvcc
SRC_DIR = src
BIN_DIR = bin
TARGET = zit

all: build

build:  clean mkdir
	$(NVCC) $(SRC_DIR)/*.cu -o $(BIN_DIR)/$(TARGET)

clean: 
	$(RM) -r $(BIN_DIR)

mkdir: 
	mkdir -p $(BIN_DIR)

run:
	./$(BIN_DIR)/$(TARGET)
