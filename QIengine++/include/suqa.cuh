#pragma once
#include <iostream>
#include <stdio.h>
#include <vector>
#include "complex_defines.cuh"


//const double TWOSQINV = 1./sqrt(2.);
#define TWOSQINV (1./sqrt(2.0))

namespace suqa{

extern uint blocks, threads;

/* Utility procedures */
double vnorm(const ComplexVec& v);
void vnormalize(ComplexVec& v);

template<typename T>
void apply_2x2mat(T& x1, T& x2, const T& m11, const T& m12, const T& m21, const T& m22){

            T x1_next = m11 * x1 + m12 * x2;
            T x2_next = m21 * x1 + m22 * x2;
            x1 = x1_next;
            x2 = x2_next;
}


/* SUQA gates */
//
void apply_x(ComplexVec& state, uint q);
void apply_x(ComplexVec& state, const std::vector<uint>& qs);

void apply_h(ComplexVec& state, uint q);
void apply_h(ComplexVec& state, const std::vector<uint>& qs);

//void apply_cx(ComplexVec& state, uint q_control, uint q_target);
void apply_cx(ComplexVec& state, const uint& q_control, const uint& q_target, const uint& q_mask=1U);

void apply_mcx(ComplexVec& state, const std::vector<uint>& q_controls, const uint& q_target);
void apply_mcx(ComplexVec& state, const std::vector<uint>& q_controls, const std::vector<uint>& q_mask, const uint& q_target);

void apply_swap(ComplexVec& state, const uint& q1, const uint& q2);


/* SUQA utils */
void measure_qbit(ComplexVec& state, const uint& q, uint& c, const double& rdoub);

void apply_reset(ComplexVec& state, const uint& q, const double& rdoub);
void apply_reset(ComplexVec& state, const std::vector<uint>& qs, const std::vector<double>& rdoubs);

};
