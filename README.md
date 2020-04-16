# Simulator for Universal Quantum Algorithms (SUQA)
version 1.5 (04/2020)

General purpose runtime library for implementing runtime quantum algorithms and hybrids quantum-classical algorithms.


## Main project: thermal methods for Quantum Information

Estimation of thermal averages using quantum information algorithms.

### Quantum Metropolis Sampling (QMS)
Implementation of the algorithm from paper: https://www.nature.com/articles/nature09770  
The QMS applied to a frustrated triangle: https://arxiv.org/abs/2001.05328

## Structure of the project:
```bash
.  
├ Makefile (for linux compilation)  
├ README.md (this file)  
├ include  
│   ├ Rand.hpp  
│   ├ complex\_defines.cuh  
│   ├ io.hpp  
│   ├ parser.hpp  
│   ├ pcg32.h  
│   ├ qms.cuh (implementation of the qms algorithm; header only)  
│   ├ suqa.cuh  
│   └ system.cuh  
├ src  
│   ├ Rand.cpp  
│   ├ io.cpp  
│   ├ qms.cu (runs the qms algorithm)  
│   ├ suqa.cu (core engine with the implementation of quantum gates)  
│   ├ system.cu (definitions system-specifics)  
│   └ test\_evolution.cu (test for evolution of the system)  
└ vs (visual studio solution and project folders)  
    ├ qms  
    ├ suqa.sln  
    └ test_evolution  
```

Each git branch represents a different system:
- master : frustrated triangle
- z2\_matter\_gauge : model with hamiltonian evolution of a gauge theory as decribed in https://arxiv.org/abs/1903.08807 
- d4-gauge : another model from the previous paper
- z2-gauge : toy model for d4-gauge with gauge group Z2

## Compiling

Linux and Windows are supported to this date.  
This code runs only on machines with NVIDIA gpus.  
(Previous versions ran also on cpu only, but it has been dismissed since it wasn't mantained anymore; maybe in the future...)

### Linux

#### dependencies
* g++ with std>=c++11  
* CUDA toolkit  
* Make  

#### compilation
In order to compile, run:
```bash
make <target>
```
where <target> can be 'release' for the qms algorithm in production, 
'debug' for qms in debug mode, 'profile' for qms in profile mode, 
or 'test\_evolution' for compiling test code for the system evolutor.

### Windows

#### dependencies
* Visual Studio 2019
* CUDA toolkit

#### compilation
The Visual Studio solution is in 'vs/suqa.sln'.  
It contains two projects, 'qms' and 'test\_evolution';
to build one of them, right-click on the project name on 'Solution Explorer', and select 'Set as Startup Project',
then select the mode of compilation 'Release' or 'Debug' on the upper bar, and right-click again on the project name selecting 'Build'.  
The executable will be created in the folder 'vs/x64/Release' or 'vs/x64/Debug' depending on the compilation mode.  
To run it, e.g., you can click on 'Tools/Command Line/Developer PowerShell' on the upper bar to open a shell.  


## Collaborators (in chronological order)
Giuseppe Clemente (giuseppe.clemente93@gmail.com)  
Marco Cardinali  
Lorenzo Maio  
Claudio Bonanno  
Riccardo Aiudi  
