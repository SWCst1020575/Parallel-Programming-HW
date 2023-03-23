CXX = mpicxx
CC  = mpicc
CPPFLAGS = -I./lodepng
LDFLAGS = -pthread -fopenmp -lm 
CXXFLAGS = -std=c++17 -O3
TARGET = hw2
HW2 = hw2.cc
LODEPNG = ./lodepng/lodepng.cpp

all: $(TARGET)


$(TARGET): $(HW2) $(LODEPNG)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(HW2) $(LODEPNG) -o $(TARGET)

.PHONY: clean
clean:
	rm -r hw2	
