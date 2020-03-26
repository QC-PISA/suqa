.PHONY: release debug release_gpu debug_gpu profile_gpu clean 

CXX = g++
NVCC = /usr/local/cuda-9.1/bin/nvcc
CXXFLAGS = -Wall -Wextra -std=c++11 -I./include -I.
NVCCFLAGS = -std=c++11 -I. -I./include -DCUDA -lcudart
INCLUDES = $(wildcard include/*)

SRC = src
OBJDIR = obj

release_gpu: NVCCFLAGS += -O3 -DNDEBUG
release_gpu: main_gpu

debug_gpu: NVCCFLAGS += -O3 -DNDEBUG
debug_gpu: main_gpu

profile_gpu: NVCCFLAGS += -m64 -O3 -G -g -DNDEBUG
profile_gpu: main_gpu

release: CXXFLAGS += -O3 -DNDEBUG
release: main
	
debug: CXXFLAGS += -g
debug: main


$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/%.cpp.o: $(SRC)/%.cpp $(INCLUDES) $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

$(OBJDIR)/%.cu.o: $(SRC)/%.cu $(INCLUDES) $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< 
	
main: $(OBJDIR)/main.cpp.o $(OBJDIR)/system.cpp.o $(OBJDIR)/suqa.cpp.o 
	$(CXX) $(CXXFLAGS) $(OBJDIR)/main.cpp.o $(OBJDIR)/system.cpp.o $(OBJDIR)/suqa.cpp.o  -o main

main_gpu: NVCCFLAGS += -DNDEBUG
main_gpu: $(OBJDIR)/main.cu.o $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/io.cpp.o
	$(NVCC) $^ -o $@

test_evolution: NVCCFLAGS += -DNDEBUG
test_evolution: $(OBJDIR)/test_evolution.cu.o $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/io.cpp.o
	$(NVCC) $^ -o $@

z2_matter_test_evolution: NVCCFLAGS += -DNDEBUG
z2_matter_test_evolution:  $(OBJDIR)/z2_test_evolution.cu.o $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/io.cpp.o
	$(NVCC)  $^ -o $@


clean:
	rm -rf main main_gpu $(OBJDIR)

