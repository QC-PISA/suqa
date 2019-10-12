.PHONY: release debug clean 

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

release: CXXFLAGS += -O3 -DNDEBUG
release: main
	
debug: CXXFLAGS += -g
debug: main
	
main: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o main

clean:
	rm -rf main

