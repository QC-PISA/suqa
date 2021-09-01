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

const int syst_qbits = 3;      // number of system's qubit 

const bmReg bm_spin={0,1,2};    
//extern double g_beta;


void init_state();

void evolution(const double& t, const int& n);

void apply_C(const uint &Ci);
void apply_C_inverse(const uint &Ci);

std::vector<double> get_C_weigthsums();

// qsa specifics
const bmReg bm_spin_tilde={3,4,5};
void qsa_init_state();
void evolution_szegedy(const double& t, const int& n);
void evolution_measure(const double& t, const int& n);
void evolution_tracing(const double& t, const int& n);

void qsa_apply_C(const uint &Ci);
void qsa_apply_C_inverse(const uint &Ci);
