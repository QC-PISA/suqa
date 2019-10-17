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

/* Hamiltonian
 *
 * H = eps {{1, 0, 0}, {0, 1, 1}, {0, 1, 1}}
 *
 */


// simulation parameters
double beta;
double eps;
double f1;
double f2;
const uint nqubits = 7;
const uint Dim = (uint)pow(2.0, nqubits); // simulation hyperparameters
uint max_reverse_attempts;
uint metro_steps;
uint reset_each;
unsigned long long iseed = 0ULL;

uint gCi;
uint c_acc = 0;


// Global state of the system.
// Ordering (less to most significant)
// psi[0], psi[1], E_old[0], E_old[1], E_new[0], E_new[1], acc //, qaux[0]
vector<Complex> gState(Dim,0.0);

//vector<double> energy_measures;
vector<double> X_measures;
vector<double> E_measures;

// Operator X parameter
const double phi = (1.+sqrt(5.))/2.;
const double mphi_inv = -1./phi;
const double Sa = phi/sqrt(2.+phi);
const double Sb = 1/sqrt(2.+phi);
const double S_10=Sa, S_12=Sb, S_20=-Sb, S_22=Sa;
const double twosqinv = 1./sqrt(2.);


// constants
const Complex rphase_m[3] = {exp((2*M_PI)*iu), exp((2*M_PI/2.)*iu), exp((2*M_PI/4.)*iu)};

// Utilities

pcg rangen;

// bit masks
enum bm_idxs {  bm_psi0, 
                bm_psi1,
                bm_E_old0,
                bm_E_old1,
                bm_E_new0,
                bm_E_new1,
                bm_acc};


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

void check_unused(){
    uint mask1 = 3U;
    uint mask2 = 12U;
    for(uint i = 0U; i < gState.size(); ++i){
        if( (i & mask1) == mask1){
            assert(norm(gState[i])<1e-8);
        }
        if( (i & mask2) == mask2){
            assert(norm(gState[i])<1e-8);
        }
    } 
}

void reset_non_state_qbits(){
    DEBUG_CALL(cout<<"\n\nBefore reset"<<endl);
    DEBUG_CALL(sparse_print(gState));
    qi_reset(gState, {bm_E_old0, bm_E_old1, bm_E_new0, bm_E_new1, bm_acc});
    DEBUG_CALL(cout<<"\n\nAfter reset"<<endl);
    DEBUG_CALL(sparse_print(gState));
}


void measure_qbit(vector<Complex>& state, const uint& q, uint& c){
    double prob1 = 0.0;

    for(uint i = 0U; i < state.size(); ++i){
        if((i >> q) & 1U){
            prob1+=norm(state[i]); 
        }
    }
    c = (uint)(rangen.doub() < prob1); // prob1=1 -> c = 1 surely
    
    if(c){ // set to 0 coeffs with bm_acc 0
        for(uint i = 0U; i < state.size(); ++i){
            if(((i >> q) & 1U) == 0U)
                state[i] = {0.0, 0.0};        
        }
    }else{ // set to 0 coeffs with bm_acc 1
        for(uint i = 0U; i < state.size(); ++i){
            if(((i >> q) & 1U) == 1U)
                state[i] = {0.0, 0.0};        
        }
    }
    vnormalize(state);
}

//TODO: can be optimized for multiple qbits measures?
void measure_qbits(vector<Complex>& state, const vector<uint>& qs, vector<uint>& cs){
    for(uint k = 0U; k < qs.size(); ++k)
        measure_qbit(state, qs[k], cs[k]);
}

void apply_Phi(){
   // quantum phase estimation (here trivial)
    DEBUG_CALL(cout<<"\nApply Phi old\n"<<endl);
	uint mask = 3U;
	for(uint i_0 = 0U; i_0 < gState.size(); ++i_0){
        if((i_0 & mask) == 0U){
            uint i_1 = i_0 | 1U;
            uint i_2 = i_0 | 2U;

            Complex a_0 = gState[i_0];
            Complex a_1 = gState[i_1];
            Complex a_2 = gState[i_2];
            
            gState[i_0] = a_0;
            gState[i_1] = twosqinv*a_1 - twosqinv*a_2;
            gState[i_2] = twosqinv*a_1 + twosqinv*a_2;
        }
    }
    DEBUG_CALL(cout<<"\napply S:\n"<<endl);
    DEBUG_CALL(sparse_print(gState));


    qi_cx(gState, bm_psi1, bm_E_old1);

    DEBUG_CALL(cout<<"\napply Phi_diag old:\n"<<endl);
    DEBUG_CALL(sparse_print(gState));

	for(uint i_0 = 0U; i_0 < gState.size(); ++i_0){
        if((i_0 & mask) == 0U){
            uint i_1 = i_0 | 1U;
            uint i_2 = i_0 | 2U;

            Complex a_0 = gState[i_0];
            Complex a_1 = gState[i_1];
            Complex a_2 = gState[i_2];
            
            gState[i_0] = a_0;
            gState[i_1] = twosqinv*a_1 + twosqinv*a_2;
            gState[i_2] = -twosqinv*a_1 + twosqinv*a_2;
        }
    }
    DEBUG_CALL(cout<<"\napply S_dag old:\n"<<endl);
    DEBUG_CALL(sparse_print(gState));
}

void apply_Phi_inverse(){
   // quantum phase estimation (here trivial)
    DEBUG_CALL(cout<<"\nApply Phi old inverse\n"<<endl);
	uint mask = 3U;
	for(uint i_0 = 0U; i_0 < gState.size(); ++i_0){
        if((i_0 & mask) == 0U){
            uint i_1 = i_0 | 1U;
            uint i_2 = i_0 | 2U;

            Complex a_0 = gState[i_0];
            Complex a_1 = gState[i_1];
            Complex a_2 = gState[i_2];
            
            gState[i_0] = a_0;
            gState[i_1] = twosqinv*a_1 - twosqinv*a_2;
            gState[i_2] = twosqinv*a_1 + twosqinv*a_2;
        }
    }
    DEBUG_CALL(cout<<"\napply S:\n"<<endl);
    DEBUG_CALL(sparse_print(gState));

    qi_cx(gState, bm_psi1, bm_E_old1);

    DEBUG_CALL(cout<<"\napply Phi_diag old inverse:\n"<<endl);
    DEBUG_CALL(sparse_print(gState));

	for(uint i_0 = 0U; i_0 < gState.size(); ++i_0){
        if((i_0 & mask) == 0U){
            uint i_1 = i_0 | 1U;
            uint i_2 = i_0 | 2U;

            Complex a_0 = gState[i_0];
            Complex a_1 = gState[i_1];
            Complex a_2 = gState[i_2];
            
            gState[i_0] = a_0;
            gState[i_1] = twosqinv*a_1 + twosqinv*a_2;
            gState[i_2] = -twosqinv*a_1 + twosqinv*a_2;
        }
    }
    DEBUG_CALL(cout<<"\napply S_dag old inverse:\n"<<endl);
    DEBUG_CALL(sparse_print(gState));
}

struct evolution{
    vector<uint> state_qbits;


};

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
//            uint i_3 = i_0 | 3U;

            Complex a_0 = state[i_0];
            Complex a_1 = state[i_1];
            Complex a_2 = state[i_2];
//           Complex a_3 = state[i_3];
            

            state[i_0] = exp(-dt*iu)*a_0;
            state[i_1] = exp(-dt*iu)*(cos(dt)*a_1 -sin(dt)*iu*a_2);
            state[i_2] = exp(-dt*iu)*(-sin(dt)*iu*a_1 + cos(dt)*a_2);
//            state[i_3] = a_3;
        }
    }

}

void qi_qft(vector<Complex>& state, const vector<uint>& qact){
    if(qact.size()!=2)
        throw "ERROR: qft(inverse) not implemented for nqubits != 2";

    qi_h(state, qact[1]);
    qi_crm(state, qact[0], qact[1], -2);
    qi_h(state, qact[0]);

}

void qi_qft_inverse(vector<Complex>& state, const vector<uint>& qact){

    qi_h(state, qact[0]);
    qi_crm(state, qact[0], qact[1], 2);
    qi_h(state, qact[1]);

}

void apply_phase_estimation(vector<Complex>& state, const vector<uint>& q_state, const vector<uint>& q_target, const double& t, const uint& n){
    DEBUG_CALL(cout<<"apply_phase_estimation()"<<endl);

    qi_h(state,q_target);

    // apply CUs
    double dt = t/(double)n;

    for(int trg = q_target.size() - 1; trg > -1; --trg){
        for(uint ti = 0; ti < n; ++ti){
            for(uint itrs = 0; itrs < q_target.size()-trg; ++itrs){
                qi_cu_on2(state, dt, q_target[trg], q_state);
            }
        }
    }
    
    // apply QFT^{-1}
    qi_qft_inverse(state, q_target); 

}

void apply_phase_estimation_inverse(vector<Complex>& state, const vector<uint>& q_state, const vector<uint>& q_target, const double& t, const uint& n){
    DEBUG_CALL(cout<<"apply_phase_estimation_inverse()"<<endl);

    // apply QFT
    qi_qft(state, q_target); 

    // apply CUs
    double dt = t/(double)n;

    for(uint trg = 0; trg < q_target.size(); ++trg){
        for(uint ti = 0; ti < n; ++ti){
            for(uint itrs = 0; itrs < q_target.size()-trg; ++itrs){
                qi_cu_on2(state, -dt, q_target[trg], q_state);
            }
        }
    }
    
    qi_h(state,q_target);
}

int main(int argc, char** argv){
    if(argc < 2){
        cout<<"./phase_estimation <t> <n>"<<endl;
        exit(1);
    }

    
//    std::fill_n(gState.begin(), gState.size(), 0.0);
//    gState[0] = 1.0; 

    double t = 2.*atan(1.)*stod(argv[1]);
    uint n = (uint)atoi(argv[2]);
    cout<<"Test phase estimation\n"<<endl;
    vector<Complex> state(16,0.0);

    cout<<"\nCase 0:"<<endl;
    state[0]=1.0;
    cout<<"before:"<<endl;
    sparse_print(state);
    apply_phase_estimation(state, {0, 1}, {2, 3}, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);
    cout<<"bring back (inverse pe)"<<endl;
    apply_phase_estimation_inverse(state, {0, 1}, {2, 3}, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);

    cout<<"\nCase 1:"<<endl;
    state = vector<Complex>(16, 0.0);
    state[1]=twosqinv;
    state[2]=-twosqinv;

    cout<<"before:"<<endl;
    sparse_print(state);
    apply_phase_estimation(state, {0, 1}, {2, 3}, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);
    cout<<"bring back (inverse pe)"<<endl;
    apply_phase_estimation_inverse(state, {0, 1}, {2, 3}, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);

    cout<<"\nCase 2:"<<endl;
    state = vector<Complex>(16, 0.0);
    state[1]=twosqinv;
    state[2]=twosqinv;
    cout<<"before:"<<endl;
    sparse_print(state);
    apply_phase_estimation(state, {0, 1}, {2, 3}, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);
    cout<<"bring back (inverse pe)"<<endl;
    apply_phase_estimation_inverse(state, {0, 1}, {2, 3}, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);

    cout<<"\nOther case:"<<endl;
    state = vector<Complex>(16, 0.0);
    state[1]=1.0;
    cout<<"before:"<<endl;
    sparse_print(state);
    apply_phase_estimation(state, {0, 1}, {2, 3}, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);
    cout<<"bring back (inverse pe)"<<endl;
    apply_phase_estimation_inverse(state, {0, 1}, {2, 3}, t, n);
    cout<<"after:"<<endl;
    sparse_print(state);

    cout<<"all fine :)\n"<<endl;

//    FILE * fil = fopen(outfilename.c_str(), "w");
//
//    fprintf(fil, "# it E X\n");
//
//    for(uint ei = 0; ei < X_measures.size(); ++ei){
//        fprintf(fil, "%d %.16lg %.16lg\n", ei, E_measures[ei], X_measures[ei]);
//    }
//    fclose(fil);

    cout<<"\n\tSuqa!\n"<<endl;

    return 0;
}
