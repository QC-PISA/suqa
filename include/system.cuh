#pragma once
#include <vector>
#include <complex>
#include <string>
#include <stdexcept>
#include "io.hpp"
#include "suqa.cuh"
#include "Rand.hpp"


//const bmReg bm_qlink0 =  {0,  1, 2};
//const bmReg bm_qlink2 =  {3,  4, 5};
//const bmReg bm_qlink3 =  {6,  7, 8};
//const bmReg bm_qaux   =  {9, 10, 11};
//const bmReg bm_qlink1 =  {12, 13, 14};

const int syst_qbits = 10;      // number of system's qubits
const bmReg bm_qlink0 =  {0,  1, 2};
const bmReg bm_qlink1 =  {3,  4, 5};
const bmReg bm_qlink2 =  {6,  7, 8};
const bmReg bm_qlinks[3]={bm_qlink0,bm_qlink1,bm_qlink2};
const bmReg bm_qaux   =  {9}; 
//TODO: remove ancillary qubit

extern double g_beta;

void init_state();

void evolution(const double& t, const int& n);

#define DEFAULT_THETA (1./sqrt(2))
void apply_C(const uint &Ci, double rot_angle=DEFAULT_THETA);
void apply_C_inverse(const uint &Ci, double rot_angle=DEFAULT_THETA);

std::vector<double> get_C_weigthsums();

// qsa specifics
//const bmReg bm_spin_tilde={3,4,5};
void qsa_init_state();
void evolution_szegedy(const double& t, const int& n);
void evolution_measure(const double& t, const int& n);
void evolution_tracing(const double& t, const int& n);

void qsa_apply_C(const uint &Ci);
void qsa_apply_C_inverse(const uint &Ci);
