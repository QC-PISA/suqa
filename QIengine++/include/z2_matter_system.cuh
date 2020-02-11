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

const bmReg bm_z2_qlink0   =  {0};
const bmReg bm_z2_qlink1   =  {1};
const bmReg bm_z2_qlink2   =  {2};
const bmReg bm_z2_qferm0   =  {3};
const bmReg bm_z2_qferm1   =  {4};
const bmReg bm_z2_qferm2   =  {5};
const bmReg bm_z2_qferm3   =  {6};

extern double m_mass;

void init_state(ComplexVec& state, uint Dim);

void apply_lamm_operator(ComplexVec& state);

void apply_mass_evolution(ComplexVec& state, uint q, double theta);
void apply_mass_evolution(ComplexVec& state, const bmReg& qs, double theta);

void apply_gauge_link_evolution(ComplexVec& state, uint q, double theta);
void apply_gauge_link_evolution(ComplexVec& state, const bmReg& qs, double theta);

void apply_hopping_evolution_x(ComplexVec& state, uint qlink, uint qferm_m, uint qferm_p, double theta);
