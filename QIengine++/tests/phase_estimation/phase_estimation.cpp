#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
#include <bits/stdc++.h>
#include <cmath>
#include <cassert>
#include "../../include/Rand.hpp"

#ifndef NDEBUG
    #define DEBUG_CALL(x) x
#else
    #define DEBUG_CALL(x)
#endif

using namespace std;

typedef complex<double> Complex;
const Complex iu(0, 1);


// Utilities
vector<Complex> rphase_m;

void fill_rphase(const uint& nlevels){
    rphase_m.resize(nlevels);
    uint c=1;
    for(uint i=0; i<nlevels; ++i){
        rphase_m[i] = exp((2.0*M_PI/(double)c)*iu);
        c<<=1;
    }
}

/* Hamiltonian
 *
 * H = eps {{0.5, 0, 0}, {0, 1, 1}, {0, 1, 1}}
 *
 */
const double twosqinv = 1./sqrt(2.);


// Utilities
std::ostream& operator<<(std::ostream& s, const Complex& c){
    s<<"("<<real(c)<<", "<<imag(c)<<")";
    return s;
}

template<class T>
void print(vector<T> v){
    for(const auto& el : v)
        cout<<el<<" ";
    cout<<endl;
}

void sparse_print(vector<Complex> v){
    for(uint i=0; i<v.size(); ++i){
        if(norm(v[i])>1e-8)
            cout<<"i="<<i<<" -> "<<v[i]<<"; ";
    }
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
            state[i_0] = twosqinv*(a_0+a_1);
            state[i_1] = twosqinv*(a_0-a_1);
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

// Simulation procedures

//void measure_qbit(vector<Complex>& state, const uint& q, uint& c){
//    double prob1 = 0.0;
//
//    for(uint i = 0U; i < state.size(); ++i){
//        if((i >> q) & 1U){
//            prob1+=norm(state[i]); 
//        }
//    }
//    c = (uint)(rangen.doub() < prob1); // prob1=1 -> c = 1 surely
//    
//    if(c){ // set to 0 coeffs with bm_acc 0
//        for(uint i = 0U; i < state.size(); ++i){
//            if(((i >> q) & 1U) == 0U)
//                state[i] = {0.0, 0.0};        
//        }
//    }else{ // set to 0 coeffs with bm_acc 1
//        for(uint i = 0U; i < state.size(); ++i){
//            if(((i >> q) & 1U) == 1U)
//                state[i] = {0.0, 0.0};        
//        }
//    }
//    vnormalize(state);
//}

//TODO: can be optimized for multiple qbits measures?
//void measure_qbits(vector<Complex>& state, const vector<uint>& qs, vector<uint>& cs){
//    for(uint k = 0U; k < qs.size(); ++k)
//        measure_qbit(state, qs[k], cs[k]);
//}
void qi_crm(vector<Complex>& state, const uint& q_control, const uint& q_target, const int& m){
    for(uint i = 0U; i < state.size(); ++i){
        // for the swap, not only q_target:1 but also q_control:1
        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
            state[i] *= (m>0) ? rphase_m[m] : conj(rphase_m[-m]);
        }
    }
}

void qi_cu_on2(vector<Complex>& state, const double& dt, const uint& q_control, const vector<uint>& qstate){
    uint cmask = (1U << q_control);
	uint mask = cmask; // (1U << qstate[0]) | (1U << qstate[0])
    for(const auto& qs : qstate){
        mask |= (1U << qs);
    }

	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
        if((i_0 & mask) == cmask){
      
            uint i_1 = i_0 | (1U << qstate[0]);
            uint i_2 = i_0 | (1U << qstate[1]);
            uint i_3 = i_1 | i_2;

            Complex a_0 = state[i_0];
            Complex a_1 = state[i_1];
            Complex a_2 = state[i_2];
            Complex a_3 = state[i_3];
            

            state[i_0] = exp(-dt*iu)*a_0;
            state[i_1] = exp(-dt*iu)*(cos(dt)*a_1 -sin(dt)*iu*a_2);
            state[i_2] = exp(-dt*iu)*(-sin(dt)*iu*a_1 + cos(dt)*a_2);
            state[i_3] = exp(-dt*iu)*a_3;
        }
    }

}


void qi_qft(vector<Complex>& state, const vector<uint>& qact){
    int qsize = qact.size();

    for(int outer_i=qsize-1; outer_i>=0; outer_i--){
        qi_h(state, qact[outer_i]);
        for(int inner_i=outer_i-1; inner_i>=0; inner_i--){
            qi_crm(state, qact[inner_i], qact[outer_i], -1-(outer_i-inner_i));
        }
    }
//    es: qsize=2
//    o=1,i=0 -> 2
//    o=0, x
//    es: qsize=3
//    h(2)
//    o=2,i=1 -> 2
//    o=2,i=0 -> 3
//    h(1)
//    o=1,i=0 -> 2
//    h(0)
//    o=0, x
//    qi_h(state, qact[1]);
//    qi_crm(state, qact[0], qact[1], -2);
//    qi_h(state, qact[0]);

    
}

void qi_qft_inverse(vector<Complex>& state, const vector<uint>& qact){

//    qi_h(state, qact[0]);
//    qi_crm(state, qact[0], qact[1], 2);
//    qi_h(state, qact[1]);

    int qsize = qact.size();

    for(int outer_i=0; outer_i<qsize; outer_i++){
        for(int inner_i=0; inner_i<outer_i; inner_i++){
            qi_crm(state, qact[inner_i], qact[outer_i], 1+(outer_i-inner_i));
        }
        qi_h(state, qact[outer_i]);
    }
    // es: qact.size()=2
    // o=0, x
    // h(0)
    // o=1,i=0 -> 2
    // h(1)
    //
    // es: qact.size()=3
    // o=0, x
    // h(0)
    // o=1,i=0 -> 2
    // o=1, x
    // h(1)
    // o=2,i=0 -> 3
    // o=2,i=1 -> 2
    // o=2, x 
    // h(2)
}

void apply_phase_estimation(vector<Complex>& state, const vector<uint>& q_state, const vector<uint>& q_target, const double& t, const uint& n){
    (void)n;
   
    DEBUG_CALL(cout<<"apply_phase_estimation()"<<endl);

    qi_h(state,q_target);

//    cout<<"after h"<<endl;
//    sparse_print(state);

    // apply CUs
//    double dt = t/(double)n;

    for(int trg = q_target.size() - 1; trg > -1; --trg){
//        for(uint ti = 0; ti < n; ++ti){
        qi_cu_on2(state, pow(2,q_target.size()-1-trg)*t, q_target[trg], q_state);
//            for(uint itrs = 0; itrs < (uint)pow(2,q_target.size()-trg); ++itrs){
//                qi_cu_on2(state, dt, q_target[trg], q_state);
//            }
//        }
    }
    cout<<"after CUs"<<endl;
    sparse_print(state);
    
    // apply QFT^{-1}
    qi_qft_inverse(state, q_target); 

}

void apply_phase_estimation_inverse(vector<Complex>& state, const vector<uint>& q_state, const vector<uint>& q_target, const double& t, const uint& n){
    (void)n;
    if(q_state.size()!=2){
        throw std::runtime_error("ERROR: state with more than 2 qbits are still not supported");
    }
    DEBUG_CALL(cout<<"apply_phase_estimation_inverse()"<<endl);

    // apply QFT
    qi_qft(state, q_target); 

//    sparse_print(state);

    // apply CUs
//    double dt = t/(double)n;

    for(uint trg = 0; trg < q_target.size(); ++trg){
//        for(uint ti = 0; ti < n; ++ti){
        qi_cu_on2(state, -pow(2,q_target.size()-1-trg)*t, q_target[trg], q_state);
//            for(uint itrs = 0; itrs < q_target.size()-trg; ++itrs){
//                qi_cu_on2(state, -dt, q_target[trg], q_state);
//            }
//        }
    }
    
    qi_h(state,q_target);
}

int main(int argc, char** argv){
    if(argc < 4){
        cout<<"./phase_estimation <t> <n> <k>"<<endl;
        exit(1);
    }

    
//    std::fill_n(gState.begin(), gState.size(), 0.0);
//    gState[0] = 1.0; 

    double t = 2.*atan(1.)*stod(argv[1]);
    uint n = (uint)atoi(argv[2]);
    uint k = (uint)atoi(argv[3]);
    cout<<"Test phase estimation\n"<<endl;
    
    uint totsdim = (uint)pow(2,2+k);
    vector<Complex> state(totsdim,0.0);

    vector<uint> enestates(k);
    for(uint ki=0; ki<k; ++ki){
        enestates[ki]=2+ki;
    }

    state = vector<Complex>(totsdim, 0.0);
    state[0]=1.0;
    cout<<"before:"<<endl;
    sparse_print(state);
    apply_phase_estimation(state, {0, 1}, enestates, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);
    cout<<"bring back (inverse pe)"<<endl;
    apply_phase_estimation_inverse(state, {0, 1}, enestates, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);

//    cout<<"\nCase 1:"<<endl;
//    state = vector<Complex>(totsdim, 0.0);
//    state[1]=twosqinv;
//    state[2]=-twosqinv;
//
//    cout<<"before:"<<endl;
//    sparse_print(state);
//    apply_phase_estimation(state, {0, 1}, enestates, t, n);
//    cout<<"after:"<<endl;
//    sparse_print(state);
//    cout<<"bring back (inverse pe)"<<endl;
//    apply_phase_estimation_inverse(state, {0, 1}, enestates, t, n);
//    cout<<"after:"<<endl;
//    sparse_print(state);
//
//    cout<<"\nCase 2:"<<endl;
//    state = vector<Complex>(16, 0.0);
//    state = vector<Complex>(totsdim, 0.0);
//    state[1]=twosqinv;
//    state[2]=twosqinv;
//    cout<<"before:"<<endl;
//    sparse_print(state);
//    apply_phase_estimation(state, {0, 1}, {2, 3}, t, n);
//    cout<<"after:"<<endl;
//    sparse_print(state);
//    cout<<"bring back (inverse pe)"<<endl;
//    apply_phase_estimation_inverse(state, {0, 1}, {2, 3}, t, n);
//    apply_phase_estimation(state, {0, 1}, enestates, t, n);
//    cout<<"after:"<<endl;
//    sparse_print(state);
//    cout<<"bring back (inverse pe)"<<endl;
//    apply_phase_estimation_inverse(state, {0, 1}, enestates, t, n);
//    cout<<"after:"<<endl;
//    sparse_print(state);
//
//    cout<<"\nOther case:"<<endl;
//    state = vector<Complex>(16, 0.0);
//    state[1]=1.0;
//    cout<<"before:"<<endl;
//    sparse_print(state);
//    apply_phase_estimation(state, {0, 1}, {2, 3}, t, n);
//    cout<<"after:"<<endl;
//    sparse_print(state);
//    cout<<"bring back (inverse pe)"<<endl;
//    apply_phase_estimation_inverse(state, {0, 1}, {2, 3}, t, n);
//    state = vector<Complex>(totsdim, 0.0);
//    state[1]=1.0;
//    cout<<"before:"<<endl;
//    sparse_print(state);
//    apply_phase_estimation(state, {0, 1}, enestates, t, n);
//    cout<<"after:"<<endl;
//    sparse_print(state);
//    cout<<"bring back (inverse pe)"<<endl;
//    apply_phase_estimation_inverse(state, {0, 1}, enestates, t, n);
//    cout<<"after:"<<endl;
//    sparse_print(state);
//
    cout<<"all fine :)\n"<<endl;

//    FILE * fil = fopen(outfilename.c_str(), "w");
//
//    fprintf(fil, "# it E X\n");
//
//    for(uint ei = 0; ei < X_measures.size(); ++ei){
//        fprintf(fil, "%d %.32lg %.16lg\n", ei, E_measures[ei], X_measures[ei]);
//    }
//    fclose(fil);

    cout<<"\n\tSuqa!\n"<<endl;

    return 0;
}
