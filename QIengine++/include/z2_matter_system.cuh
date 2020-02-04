#pragma once
#include <vector>
#include <complex>
#include <string>
#include <stdexcept>
#include "io.hpp"
#include "suqa.cuh"


//const bmReg bm_qlink0 =  {0,  1, 2};
//const bmReg bm_qlink1 =  {3,  4, 5};
//const bmReg bm_qlink2 =  {6,  7, 8};
//const bmReg bm_qlink3 =  {9, 10, 11};
//const bmReg bm_qaux   =  {12};

extern double m_mass;

void init_state(ComplexVec& state, uint Dim);


