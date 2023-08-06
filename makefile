CXXARGS = -std=c++14 -g -ICUDASAMPLES/Common -gencode arch=compute_60,code=sm_60
UNAME = $(shell uname)

ifeq ($(UNAME), Linux)
LIBS = -lcuda -lstdc++fs
else
LIBS = -lcuda
endif

all: capture.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) analyze.cu -o all

