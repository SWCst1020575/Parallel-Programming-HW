CXX = g++
CXXFLAGS = -std=c++17 -O3 -pthread -fopenmp
SOURCE = hw1.cc Sokoban.cc Step.cc
TARGETS = hw1

	
.PHONY: all
all: 
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGETS)
	

.PHONY: clean
clean:
	rm -f $(TARGETS)
