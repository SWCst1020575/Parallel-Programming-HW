NVFLAGS := -std=c++11 -O3 -Xptxas=-v -arch=sm_61 -gencode arch=compute_61,code=compute_61 -lcudart
TARGET1 := hw6-1
TARGET2 := hw6-2
TARGET3 := hw6-3
TARGET4 := hw6-4

.PHONY: all
all: $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET1): hw6-1.cu
	nvcc hw6-1.cu -o hw6-1 $(NVFLAGS)

$(TARGET2): hw6-2.cu
	nvcc hw6-2.cu -o hw6-2 $(NVFLAGS)

$(TARGET3): hw6-3.cu
	nvcc hw6-3.cu -o hw6-3 $(NVFLAGS)

clean:
	rm -rf hw6-1 hw6-2 hw6-3 *.o _* *.txt *.log *.ptx *.ptxas
