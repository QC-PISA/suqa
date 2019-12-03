.PHONY: release debug clean 

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11
INCLUDES = $(wildcard include/*)

SRC = src
OBJDIR = obj

release: CXXFLAGS += -O3 -DNDEBUG
release: main
	
debug: CXXFLAGS += -g
debug: main

$(OBJDIR)/%.o: $(SRC)/%.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) -c -o $@ $< 
	
# evolution.o: $(SRC)/evolution.cpp $(INCLUDES)
# 	$(CXX) $(CXXFLAGS) -c $(SRC)/evolution.cpp -o $(OBJDIR)/evolution.o
# 
# main.o: $(SRC)/main.cpp $(INCLUDES)
# 	$(CXX) $(CXXFLAGS) -c $(SRC)/main.cpp

main: $(OBJDIR)/evolution.o $(OBJDIR)/main.o
	$(CXX) $(CXXFLAGS) $(OBJDIR)/evolution.o $(OBJDIR)/main.o -o main


clean:
	rm -rf main $(OBJDIR)/*.o
