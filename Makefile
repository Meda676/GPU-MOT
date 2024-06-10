ifdef DEBUG
GCC_DEBUGFLAGS = -g -O0
NVCC_DEBUGFLAG = -g -G -O0
else
NVCC_DEBUGFLAG = -O3
GCC_DEBUGFLAGS = -O3
endif

ifeq (,$(filter clean,$(MAKECMDGOALS)))
ifndef MAX_KF
$(error MAX_KF is not set)
endif
ifeq ($(shell test $(MAX_KF) -gt 1000; echo $$?),0)
$(error MAX_KF must be <= 1000)
endif
endif

CUDALIB_ = /usr/local/cuda/lib
CUDALIB = $(CUDALIB_)
OS_ARCH = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")

ifeq ($(OS_ARCH),64)
CUDALIB = $(CUDALIB_)64
endif

CUDA = /usr/local/cuda/include
NVCC = nvcc
NVCCFLAGS = -std=c++17 -x cu -dc \
	-DMAX_KF=$(MAX_KF) --compiler-options '$(GCC_DEBUGFLAGS)' --compiler-options '-pthread' --compiler-options '-mavx2' --compiler-options '-mfma' --compiler-options '-march=native' \
	--expt-relaxed-constexpr $(NVCC_DEBUGFLAG) $(GENCOD_FLAGS) -I. -I$(CUDA) -I$(EIGEN3) -I$(OPENCV4)
SOURCES = $(shell find ./src -name "*.cpp")
OBJECTS = $(patsubst %.cpp, %.o, $(SOURCES))
BUILDDIR = .
LDFLAGS = -L$(PHAST_DIR)/lib -L$(CUDALIB) -lcudart -lcurand -lcublas -lopencv_core -lopencv_highgui -lopencv_imgproc
GENCODE_FLAGS := -gencode arch=compute_89,code=sm_89 -gencode arch=compute_87,code=sm_87 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_72,code=sm_72 \
	-gencode arch=compute_62,code=sm_62 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_53,code=sm_53 

all: gpu_mot

gpu_mot: $(OBJECTS) 
	$(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
ifndef EIGEN3
	$(error 'please define an environment variable EIGEN3 with Eigen3 include path')
endif
ifndef OPENCV4
	$(error 'please define an environment variable OPENCV4 with OpenCV4 include path')
endif
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf src/*.o gpu_mot
