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

cudatest:
	nvcc -g -O3 -I include -DCUDA -c src/suqa.cu -o obj/suqa.o -lcudart
	nvcc -g -O3 -I include -DCUDA -c src/cuda_tester.cu -o obj/cuda_tester.o -lcudart
	nvcc obj/cuda_tester.o obj/suqa.o -o cudatest

cudatest_host:
	nvcc -g -O3 -I include -DCUDA -DCUDA_HOST -c src/suqa.cu -o obj/suqa.o -lcudart
	nvcc -g -O3 -I include -DCUDA -DCUDA_HOST -c src/cuda_tester.cu -o obj/cuda_tester.o -lcudart
	nvcc obj/cuda_tester.o obj/suqa.o -o cudatest_host

clean:
	rm -rf main main_gpu $(OBJDIR)/*.o cudatest cudatest_host 

