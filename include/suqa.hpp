#pragma once
#include <iostream>
#include <complex>
#include <stdio.h>
#include <vector>

using namespace std;

typedef complex<double> Complex;
const Complex iu(0, 1);

const double TWOSQINV = 1./sqrt(2.);


namespace suqa{

/* Utility procedures */
double vnorm(const vector<Complex>& v);
void vnormalize(vector<Complex>& v);

template<typename T>
void apply_2x2mat(T& x1, T& x2, const T& m11, const T& m12, const T& m21, const T& m22){

            T x1_next = m11 * x1 + m12 * x2;
            T x2_next = m21 * x1 + m22 * x2;
            x1 = x1_next;
            x2 = x2_next;
}

/* SUQA gates */

void qi_reset(vector<Complex>& state, const uint& q);
void qi_reset(vector<Complex>& state, const vector<uint>& qs);

void qi_x(vector<Complex>& state, const uint& q);
void qi_x(vector<Complex>& state, const vector<uint>& qs);

void qi_h(vector<Complex>& state, const uint& q);
void qi_h(vector<Complex>& state, const vector<uint>& qs);

void qi_cx(vector<Complex>& state, const uint& q_control, const uint& q_target);
void qi_cx(vector<Complex>& state, const uint& q_control, const uint& q_mask, const uint& q_target);

void qi_mcx(vector<Complex>& state, const vector<uint>& q_controls, const uint& q_target);
void qi_mcx(vector<Complex>& state, const vector<uint>& q_controls, const vector<uint>& q_mask, const uint& q_target);

void qi_swap(vector<Complex>& state, const uint& q1, const uint& q2);

};
