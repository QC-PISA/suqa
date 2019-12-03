.PHONY: release debug clean 

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11
INCLUDES = $(wildcard include/*)
SRC = src

release: CXXFLAGS += -O3 -DNDEBUG
release: main
	
debug: CXXFLAGS += -g
debug: main
	
evolution.o: $(SRC)/evolution.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) -c $(SRC)/evolution.cpp

main.o: $(SRC)/main.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) -c $(SRC)/main.cpp

main: evolution.o main.o
	$(CXX) $(CXXFLAGS) evolution.o main.o -o main
	rm -rf evolution.o main.o


clean:
	rm -rf main

