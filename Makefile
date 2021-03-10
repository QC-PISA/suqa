.PHONY: help setvars clean 


SETVARS = 0

CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -Wextra -std=c++11 -I./include -I. -O3
NVCCFLAGS = -ccbin g++ -std=c++11 -I. -I./include -lcudart -O3
INCLUDES = $(wildcard include/*)

DEVICE ?= cpu
MODE ?= release
ACCESS ?= dense

SRC = src
OBJDIR = obj
COMPILE = $(CXX) $(CXXFLAGS)
SUQAOBJS = $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/Rand.cpp.o $(OBJDIR)/io.cpp.o

help:
	@echo "usage (the first option for each flag is the default one):"
	@echo "make <rule> [DEVICE=cpu/gpu] [MODE=release/debug/profile] [ACCESS=dense/sparse]\n"
	@echo "available rules:"
	@echo "\thelp\t\t\t- print this text"
	@echo "\ttest_suqa\t\t- some tests for the suqa gates and structures"
	@echo "\ttest_evolution\t\t- compile executable for the evolution operator written of the 'system'"
	@echo "\tqms\t\t\t- compile executable for the quantum metropolis sampling applied to 'system'"
	@echo "\tqsa\t\t\t- compile executable for the quantum-quantum sampling algorithm applied to 'system'"
	@echo "\tclean\t\t\t- clean executables and objects"

setvars:
ifeq ($(SETVARS),0)
# no support for sparse gpu access yet!
ifeq ($(DEVICE),gpu)
NVCCFLAGS += -DGPU
COMPILE = $(NVCC) $(NVCCFLAGS)
else
ifeq ($(ACCESS),sparse)
CXXFLAGS += -DSPARSE
endif
endif
ifeq ($(MODE),release)
CXXFLAGS += -DNDEBUG
NVCCFLAGS += -DNDEBUG
endif
ifeq ($(MODE),profile)
CXXFLAGS += -DNDEBUG -g 
NVCCFLAGS += -DNDEBUG -m64 -G -g
endif
SETVARS = 1
endif


$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/%.cpp.o: $(SRC)/%.cpp $(INCLUDES) $(OBJDIR) setvars
	$(COMPILE) -o $@ -c $< 

$(OBJDIR)/%.cu.o: $(SRC)/%.cu $(INCLUDES) $(OBJDIR) setvars
# need this temporary file because g++ isn't flexible and changes mode depending on the file suffix
ifeq ($(DEVICE),gpu)
	$(COMPILE) -o $@ -c $<
else
	cp $< $<_temp.cpp
	$(COMPILE) -o $@ -c $<_temp.cpp
	rm $<_temp.cpp
endif

test_suqa: $(OBJDIR)/test_suqa.cu.o $(SUQAOBJS) setvars
	$(COMPILE) $< $(SUQAOBJS) -o $@

test_evolution: $(OBJDIR)/test_evolution.cu.o $(SUQAOBJS) setvars
	$(COMPILE) $< $(SUQAOBJS) -o $@

qms: $(OBJDIR)/qms.cu.o $(SUQAOBJS) setvars
	$(NVCC) $< $(SUQAOBJS) -o $@

qsa: $(OBJDIR)/qsa.cu.o $(SUQAOBJS) setvars
	$(NVCC) $< $(SUQAOBJS) -o $@


clean:
	rm -rf test_suqa test_evolution qms qsa obj

