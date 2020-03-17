#pragma once
#include <vector>
#include <complex>
#include <string>
#include <stdexcept>
#include "io.hpp"
#include "suqa.cuh"



//const bmReg bm_qlink0 =  {0,  1, 2};
//const bmReg bm_qlink2 =  {3,  4, 5};
//const bmReg bm_qlink3 =  {6,  7, 8};
//const bmReg bm_qaux   =  {9, 10, 11};
//const bmReg bm_qlink1 =  {12, 13, 14};

const bmReg bm_qlink0 =  {0,  1, 2};
const bmReg bm_qlink1 =  {3,  4, 5};
const bmReg bm_qlink2 =  {6,  7, 8};
const bmReg bm_qlink3 =  {9, 10, 11};
const bmReg bm_qaux   =  {12};

extern double g_beta;

void init_state(ComplexVec& state, uint Dim);

void evolution(ComplexVec& state, const double& t, const int& n);

void apply_C(ComplexVec& state, const uint &Ci);

void apply_C_inverse(ComplexVec& state, const uint &Ci);

std::vector<double> get_C_weigthsums();
