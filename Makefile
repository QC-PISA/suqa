.PHONY: release debug profile clean 

CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -Wextra -std=c++11 -I./include -I.
NVCCFLAGS = -std=c++11 -I. -I./include -lcudart
INCLUDES = $(wildcard include/*)

SRC = src
OBJDIR = obj

## default rule
release: NVCCFLAGS += -O3 -DNDEBUG
release: qms

debug: NVCCFLAGS += -O3
debug: qms

profile: NVCCFLAGS += -m64 -O3 -G -g -DNDEBUG
profile: qms

$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/%.cpp.o: $(SRC)/%.cpp $(INCLUDES) $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

$(OBJDIR)/%.cu.o: $(SRC)/%.cu $(INCLUDES) $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< 
	
qms: $(OBJDIR)/qms.cu.o $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/Rand.cpp.o $(OBJDIR)/io.cpp.o
	$(NVCC) $^ -o $@

test_evolution: NVCCFLAGS += -DNDEBUG
test_evolution: $(OBJDIR)/test_evolution.cu.o $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/io.cpp.o
	$(NVCC) $^ -o $@

clean:
	rm -rf qms test_evolution $(OBJDIR) 

