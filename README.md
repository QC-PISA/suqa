# Simulator for Universal Quantum Algorithms (SUQA)
version 1.8 (07/2021)

General purpose runtime library for implementing runtime quantum algorithms and hybrids quantum-classical algorithms.


## Main project: Thermal Methods for Quantum Information

Estimation of thermal averages using quantum information algorithms.<br>
Up to now we considered two types of algorithms: QMS and QSA, discussed next.

### Quantum Metropolis Sampling (QMS)
Implementation of the QMS algorithm from paper: https://www.nature.com/articles/nature09770  <br>
The QMS applied to a frustrated triangle: https://arxiv.org/abs/2001.05328

### Quantum Metropolis Sampling (QSA)
Implementation of the QQSA algorithm from paper: https://www.pnas.org/content/109/3/754 [Implemented by Riccardo Aiudi]

## Structure of the project:
```bash
.  
├── README.md (this file)  
├── Makefile (for linux compilation)  
├── include
│   ├── complex\_defines.cuh
│   ├── io.hpp
│   ├── parser.hpp
│   ├── pcg32.h
│   ├── suqa.cuh            (prototypes of the suqa library)
│   ├── qms.cuh             (core of the qms algorithm)  
│   ├── qsa.cu              (core of the qsa algorithm)
│   ├── Rand.hpp
│   ├── suqa\_cpu.hpp       (suqa cpu core functions)
│   ├── suqa\_kernels.cuh   (suqa gpu core functions)
│   └── system.cuh          (prototypes for any system) 
├── src
│   ├── io.cpp              (input/output facilities)
│   ├── qms.cu              (runs the qms algorithm)
│   ├── qsa.cu              (runs the qsa algorithm)
│   ├── Rand.cpp            (pseudorandom number generators)
│   ├── suqa.cu             (core engine)
│   ├── system.cu           (system-specific structures and functions)
│   ├── test\_evolution.cu  (tests the system's evolution operator)
│   └── test\_suqa.cu       (tests the suqa functions and structures)
└── vs      (visual studio solution and project folders)  
    ├── qms  
    ├── suqa.sln  
    └── test\_evolution  

```

Each git branch represents a different system:
- master : frustrated triangle
- z2\_matter\_gauge : model with hamiltonian evolution of a gauge theory as decribed in https://arxiv.org/abs/1903.08807 [implemented by Marco Cardinali]
- d4-gauge : another model from the previous paper  [implemented only the evolution]
- z2-gauge : toy model for d4-gauge with gauge group Z2 [implemented by Lorenzo Maio]

## Compiling

Linux and Windows are supported to this date.  
This code runs both on cpu only or on machines with NVIDIA gpus (CUDA).

### Linux

#### dependencies
* g++ with std>=c++11  
* CUDA toolkit (if compiled for gpu devices)
* Make  

#### compilation
To know the different compilation options run
```bash
make help
```

### Windows

#### dependencies
* Visual Studio 2019
* CUDA toolkit (if compiled for gpu devices)

#### compilation
The Visual Studio solution is in 'vs/suqa.sln'.<br>
It contains two projects, 'qms' and 'test\_evolution';<br>
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
