CXX = g++
NVCC = nvcc

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = .

CXXFLAGS = -O3 -std=c++17 -I$(INC_DIR) -Wall
NVCCFLAGS = -O3 -std=c++17 -I$(INC_DIR) -arch=sm_70
OPENMP_FLAGS = -fopenmp

TARGET = $(BIN_DIR)/barnes_hut

all: build

directories:
	@mkdir -p $(BUILD_DIR)

build: $(BUILD_DIR)/main.o $(BUILD_DIR)/wrappers_cuda.o $(BUILD_DIR)/octree.o $(BUILD_DIR)/barnes_hut_serial.o $(BUILD_DIR)/barnes_hut_hybrid.o
	$(NVCC) $(NVCCFLAGS) $^ -o $(TARGET) -lgomp

$(BUILD_DIR)/main.o $(BUILD_DIR)/wrappers_cuda.o $(BUILD_DIR)/octree.o $(BUILD_DIR)/barnes_hut_serial.o $(BUILD_DIR)/barnes_hut_hybrid.o: | directories

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(OPENMP_FLAGS)" -c $< -o $@

$(BUILD_DIR)/wrappers_cuda.o: $(SRC_DIR)/wrappers_cuda.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(OPENMP_FLAGS)" -c $< -o $@

$(BUILD_DIR)/octree.o: $(SRC_DIR)/octree.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/barnes_hut_serial.o: $(SRC_DIR)/barnes_hut_serial.cpp
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(OPENMP_FLAGS)" -c $< -o $@

$(BUILD_DIR)/barnes_hut_hybrid.o: $(SRC_DIR)/barnes_hut_hybrid.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: all
	./$(TARGET) 100000 50

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all build clean run directories