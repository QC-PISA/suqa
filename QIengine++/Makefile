.PHONY: release debug clean 

CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -Wextra -std=c++11 -I./include
INCLUDES = $(wildcard include/*)

SRC = src
OBJDIR = obj

release: CXXFLAGS += -O3 -DNDEBUG
release: main
	
debug: CXXFLAGS += -g
debug: main

$(OBJDIR)/%.cpp.o: $(SRC)/%.cpp $(INCLUDES)
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

$(OBJDIR)/%.cu.o: $(SRC)/%.cu $(INCLUDES)
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $< 
	
main: $(OBJDIR)/main.cpp.o $(OBJDIR)/system.cpp.o $(OBJDIR)/suqa.cpp.o 
	$(CXX) $(CXXFLAGS) $(OBJDIR)/main.cpp.o $(OBJDIR)/system.cpp.o $(OBJDIR)/suqa.cpp.o  -o main

cudatest: src/cuda_tester.cu src/suqa.cu $(INCLUDES)
	$(NVCC) -std=c++11 -g -O3 -I include -DCUDA -c src/suqa.cu -o obj/suqa.cu.o -lcudart
	$(NVCC) -std=c++11 -g -O3 -I include -DCUDA -c src/cuda_tester.cu -o obj/cuda_tester.cu.o -lcudart
	$(NVCC) obj/cuda_tester.cu.o obj/suqa.cu.o -o cudatest

cudatest_host: src/cuda_tester.cu src/suqa.cu $(INCLUDES)
	$(NVCC) -g -O3 -I include -DCUDA -DCUDA_HOST -c src/suqa.cu -o obj/suqa.cu.o -lcudart
	$(NVCC) -g -O3 -I include -DCUDA -DCUDA_HOST -c src/cuda_tester.cu -o obj/cuda_tester.cu.o -lcudart
	$(NVCC) obj/cuda_tester.cu.o obj/suqa.cu.o -o cudatest_host

clean:
	rm -rf main main_gpu $(OBJDIR)/*.o cudatest cudatest_host 

