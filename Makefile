.PHONY: setvars clean 


SETVARS = 0

CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -Wextra -std=c++11 -I./include -I. -O3
NVCCFLAGS = -std=c++11 -I. -I./include -lcudart -O3
INCLUDES = $(wildcard include/*)
TEMPFILE = $(OBJDIR)/temp.cpp

SRC = src
OBJDIR = obj

DEVICE ?= cpu
MODE ?= release
ACCESS ?= dense
COMPILE = $(CXX) $(CXXFLAGS)

setvars:
ifeq ($(SETVARS),0)
# no support for sparse gpu access yet!
ifeq ($(DEVICE),gpu)
CXXFLAGS += -DGPU
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
	cp $< $(TEMPFILE) 
	$(COMPILE) -o $@ -c $(TEMPFILE)
	rm $(TEMPFILE)
	
qms: $(OBJDIR)/qms.cu.o $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/Rand.cpp.o $(OBJDIR)/io.cpp.o setvars
	$(NVCC) $(OBJDIR)/qms.cu.o $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/Rand.cpp.o $(OBJDIR)/io.cpp.o -o $@

test_evolution: $(OBJDIR)/test_evolution.cu.o $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/io.cpp.o setvars
	$(COMPILE) $(OBJDIR)/test_evolution.cu.o $(OBJDIR)/system.cu.o $(OBJDIR)/suqa.cu.o $(OBJDIR)/io.cpp.o -o $@

clean:
	rm -rf qms test_evolution obj

