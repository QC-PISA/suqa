.PHONY: release debug clean 

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -I./include
INCLUDES = $(wildcard include/*)

SRC = src
OBJDIR = obj

release: CXXFLAGS += -O3 -DNDEBUG
release: main
	
debug: CXXFLAGS += -g
debug: main

$(OBJDIR)/%.o: $(SRC)/%.cpp $(INCLUDES)
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ 
	
main: $(OBJDIR)/main.o $(OBJDIR)/system.o $(OBJDIR)/suqa.o 
	$(CXX) $(CXXFLAGS) $(OBJDIR)/main.o $(OBJDIR)/system.o $(OBJDIR)/suqa.o  -o main

clean:
	rm -rf main main_gpu $(OBJDIR)/*.o
