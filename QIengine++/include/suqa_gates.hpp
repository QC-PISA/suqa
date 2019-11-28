#pragma once
#include <iostream>
#include <complex>
#include <stdio.h>
#include <vector>

using namespace std;

typedef complex<double> Complex;
const Complex iu(0, 1);

const double TWOSQINV = 1./sqrt(2.);

template<class T>
void print(vector<T> v){
    for(const auto& el : v)
        cout<<el<<" ";
    cout<<endl;
}

double vnorm(const vector<Complex>& v){
    double ret = 0.0;
    for(const auto& el : v)
        ret += norm(el);
    
    return sqrt(ret);
}

void vnormalize(vector<Complex>& v){
    double vec_norm = vnorm(v);
    for(auto& el : v)
        el/=vec_norm;
}

template<class T>
void apply_2x2mat(T& x1, T& x2, const Complex& m11, const Complex& m12, const Complex& m21, const Complex& m22){

            T x1_next = m11 * x1 + m12 * x2;
            T x2_next = m21 * x1 + m22 * x2;
            x1 = x1_next;
            x2 = x2_next;
}

void qi_reset(vector<Complex>& state, const uint& q){
    for(uint i = 0U; i < state.size(); ++i){
        if((i >> q) & 1U){ // checks q-th digit in i
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            state[j]+=state[i];
            state[i]= {0.0, 0.0};
        }
    }
    vnormalize(state);
}  

void qi_reset(vector<Complex>& state, const vector<uint>& qs){
    uint mask=0U;
    for(const auto& q : qs)
        mask |= 1U << q;

    for(uint i = 0U; i < state.size(); ++i){
        if((i & mask) != 0U){ // checks q-th digit in i
            state[i & ~mask]+=state[i];
            state[i]= 0.0;
        }
    }
    vnormalize(state);
}  

void qi_x(vector<Complex>& state, const uint& q){
    for(uint i = 0U; i < state.size(); ++i){
        if((i >> q) & 1U){ // checks q-th digit in i
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            std::swap(state[i],state[j]);
        }
    }
}  

void qi_x(vector<Complex>& state, const vector<uint>& qs){
    for(const auto& q : qs)
        qi_x(state, q);
}  

void qi_h(vector<Complex>& state, const uint& q){
	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
        if((i_0 & (1U << q)) == 0U){
            uint i_1 = i_0 | (1U << q);
            Complex a_0 = state[i_0];
            Complex a_1 = state[i_1];
            state[i_0] = TWOSQINV*(a_0+a_1);
            state[i_1] = TWOSQINV*(a_0-a_1);
        }
    }
}  

void qi_h(vector<Complex>& state, const vector<uint>& qs){
    for(const auto& q : qs)
        qi_h(state, q);
}  


void qi_cx(vector<Complex>& state, const uint& q_control, const uint& q_target){
    for(uint i = 0U; i < state.size(); ++i){
        // for the swap, not only q_target:1 but also q_control:1
        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
            uint j = i & ~(1U << q_target);
            std::swap(state[i],state[j]);
        }
    }
}  
  

void qi_cx(vector<Complex>& state, const uint& q_control, const uint& q_mask, const uint& q_target){
    uint mask_qs = 1U << q_target;
    uint mask = mask_qs | (1U << q_control);
    if(q_mask) mask_qs |= (1U << q_control);
    for(uint i = 0U; i < state.size(); ++i){
        if((i & mask) == mask_qs){
            uint j = i & ~(1U << q_target);
            std::swap(state[i],state[j]);
        }
    }
}  

void qi_mcx(vector<Complex>& state, const vector<uint>& q_controls, const uint& q_target){
    uint mask = 1U << q_target;
    for(const auto& q : q_controls)
        mask |= 1U << q;

    for(uint i = 0U; i < state.size(); ++i){
        if((i & mask) == mask){
            uint j = i & ~(1U << q_target);
            std::swap(state[i],state[j]);
        }
    }
}  


void qi_mcx(vector<Complex>& state, const vector<uint>& q_controls, const vector<uint>& q_mask, const uint& q_target){
    uint mask = 1U << q_target;
    for(const auto& q : q_controls)
        mask |= 1U << q;
    uint mask_qs = 1U << q_target;
    for(uint k = 0U; k < q_controls.size(); ++k){
        if(q_mask[k]) mask_qs |= 1U << q_controls[k];
    }
    for(uint i = 0U; i < state.size(); ++i){
        if((i & mask) == mask_qs){
            uint j = i & ~(1U << q_target);
            std::swap(state[i],state[j]);
        }
    }
}  
void qi_swap(vector<Complex>& state, const uint& q1, const uint& q2){
        // swap gate: 00->00, 01->10, 10->01, 11->11
        // equivalent to cx(q1,q2)->cx(q2,q1)->cx(q1,q2)
        uint mask_q = (1U << q1);
        uint mask = mask_q | (1U << q2);
        for(uint i = 0U; i < state.size(); ++i){
            if((i & mask) == mask_q){
                uint j = (i & ~(1U << q1)) | (1U << q2);
                std::swap(state[i],state[j]);
            }
        }
}

