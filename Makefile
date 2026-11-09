# Fallback build without CMake.
#   make                 CPU build (dt_serial, dt_mpi, dt_tests) into build/
#   make CUDA=1          also compile the CUDA backend into dt_mpi
#   make CUDA=1 CUDA_ARCH=sm_90
# Run `make clean` when switching CUDA on or off.
CXX       ?= g++
MPICXX    ?= mpicxx
NVCC      ?= nvcc
CUDA_ARCH ?= sm_80
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra
BUILD     ?= build

CORE := src/dataset.cpp src/binning.cpp src/split.cpp src/tree.cpp
INC  := -Iinclude

MPI_OBJS := $(BUILD)/main_mpi.o $(BUILD)/trainer.o $(BUILD)/hist_cpu.o $(CORE:src/%.cpp=$(BUILD)/omp_%.o)
MPI_LIBS :=
ifeq ($(CUDA),1)
  MPI_OBJS += $(BUILD)/hist_cuda.o
  MPI_DEFS := -DDT_HAVE_CUDA
  CUDA_HOME ?= $(dir $(shell which $(NVCC)))..
  MPI_LIBS += -L$(CUDA_HOME)/lib64 -lcudart
  INC_CUDA := -I$(CUDA_HOME)/include
endif

all: $(BUILD)/dt_serial $(BUILD)/dt_mpi $(BUILD)/dt_tests

$(BUILD):
	mkdir -p $@

$(BUILD)/dt_serial: src/main_serial.cpp $(CORE) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas $(INC) $^ -o $@

$(BUILD)/dt_tests: tests/test_core.cpp $(CORE) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas $(INC) $^ -o $@

$(BUILD)/omp_%.o: src/%.cpp | $(BUILD)
	$(MPICXX) $(CXXFLAGS) -fopenmp $(INC) -c $< -o $@

$(BUILD)/hist_cpu.o: src/hist_cpu.cpp | $(BUILD)
	$(MPICXX) $(CXXFLAGS) -fopenmp $(INC) -c $< -o $@

$(BUILD)/trainer.o: src/trainer.cpp | $(BUILD)
	$(MPICXX) $(CXXFLAGS) -fopenmp $(INC) -c $< -o $@

$(BUILD)/main_mpi.o: src/main_mpi.cpp | $(BUILD)
	$(MPICXX) $(CXXFLAGS) -fopenmp $(MPI_DEFS) $(INC) $(INC_CUDA) -c $< -o $@

$(BUILD)/hist_cuda.o: src/hist_cuda.cu | $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -ccbin $(CXX) $(INC) -c $< -o $@

$(BUILD)/dt_mpi: $(MPI_OBJS)
	$(MPICXX) -fopenmp $^ $(MPI_LIBS) -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
