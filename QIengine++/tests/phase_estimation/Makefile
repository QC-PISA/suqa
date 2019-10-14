.PHONY: release debug clean 

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

release: CXXFLAGS += -O3 -DNDEBUG
release: phase_estimation
	
debug: CXXFLAGS += -g
debug: phase_estimation
	
phase_estimation: phase_estimation.cpp
	$(CXX) $(CXXFLAGS) phase_estimation.cpp -o phase_estimation

clean:
	rm -rf phase_estimation

