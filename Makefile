ifdef DEBUG
GCC_DEBUGFLAGS = -g -O0
NVCC_DEBUGFLAG = -g -G -O0
else
NVCC_DEBUGFLAG = -O3
GCC_DEBUGFLAGS = -O3
endif

CUDALIB_ = /usr/local/cuda/lib
CUDALIB = $(CUDALIB_)
OS_ARCH = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")

ifeq ($(OS_ARCH),64)
CUDALIB = $(CUDALIB_)64
endif

CUDA = /usr/local/cuda/include
NVCC = nvcc
NVCCFLAGS = -std=c++17 -x cu -dc --compiler-options '$(GCC_DEBUGFLAGS)' --compiler-options '-pthread' --compiler-options '-mavx2' --compiler-options '-mfma' --compiler-options '-march=native' --expt-relaxed-constexpr $(NVCC_DEBUGFLAG) $(GENCOD_FLAGS) -I. -I$(CUDA) -I/home/medaglini/Desktop/CUDA/DAT/DAT_CUDA_OKversion/lib/include/eigen3
SOURCES = $(shell find ./src -name "*.cpp")
OBJECTS = $(patsubst %.cpp, %.o, $(SOURCES))
BUILDDIR = .
LDFLAGS = -L$(PHAST_DIR)/lib -L$(CUDALIB) -lcudart -lcurand -lcublas -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
LIBS = -I/home/medaglini/Desktop/CUDA/lib/include/eigen3/ -I/usr/include/opencv2

GENCODE_SM75    := -gencode arch=compute_75,code=sm_75
GENCODE_FLAGS   := $(GENCODE_SM75)

all: data_tracker

data_tracker: $(OBJECTS) 
	$(NVCC) $(GENCODE_FLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf src/*.o data_tracker
