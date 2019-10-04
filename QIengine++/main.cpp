#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
#include <cmath>
#include "include/Rand.hpp"


using namespace std;

typedef complex<double> Complex;
const Complex iu(0, 1);

/* Hamiltonian
 *
 * H = eps diag(0, 1, 2)
 *
 */


// simulation parameters
double beta;
double eps;
double f1;
double f2;
const uint nqubits = 8;
const uint Dim = (uint)pow(2.0, nqubits);

// simulation hyperparameters
uint max_reverse_attempts;
uint metro_steps;

uint gCi;
uint c_acc = 0;


// Global state of the system.
// Ordering (less to most significant)
// psi[0], psi[1], E_old[0], E_old[1], E_new[0], E_new[1], acc, qaux[0]
vector<Complex> gState(Dim,0.0);

vector<double> energy_measures;


// Utilities

pcg rangen;

// bit masks
enum bm_idxs {  bm_psi0, 
                bm_psi1,
                bm_E_old0,
                bm_E_old1,
                bm_E_new0,
                bm_E_new1,
                bm_acc,
                bm_qaux0};


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

void qi_x(vector<Complex>& state, const uint& q){
    for(uint i = 0U; i < state.size(); ++i){
        if((i >> q) & 1U){ // checks q-th digit in i
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            std::swap(state[i],state[j]);
        }
    }
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

void qi_mcx(vector<Complex>& state, const vector<uint>& q_controls, const uint& q_target){
    uint mask = 0U;
    for(const auto& el : q_controls)
       mask |= 1U << el; 
    mask |= 1U << q_target;
    for(uint i = 0U; i < state.size(); ++i){
        if((i & mask) == mask){
            uint j = i & ~(1U << q_target);
            std::swap(state[i],state[j]);
        }
    }
    
}  

// Simulation procedures

void reset_non_state_qbits(){
    // resets E_old, E_new and acc to zero
    // sweeps all the global state vector, setting to zero
    // the amplitude for non zero values of that qubits.
    
    for(uint i = 0U; i < Dim; ++i){
        // read i in binary:
        if(i & 124U)
            gState[i] = {0.0, 0.0};   
        // equivalent (but more efficient) to this check
//                  ((i >> bm_E_old0) & 1U)
//                | ((i >> bm_E_old1) & 1U)
//                | ((i >> bm_E_new0) & 1U)
//                | ((i >> bm_E_new1) & 1U)
//                | ((i >> bm_acc) & 1U)
    }

    // normalize again
    vnormalize(gState);
}
void apply_Phi_old(){
   // quantum phase estimation (here trivial)
   qi_cx(gState, bm_psi0, bm_E_old0);
   qi_cx(gState, bm_psi1, bm_E_old1);
}

void apply_Phi_old_inverse(){
   // quantum phase estimation (here trivial)
   qi_cx(gState, bm_psi0, bm_E_old0);
   qi_cx(gState, bm_psi1, bm_E_old1);
}

void apply_Phi(){
   // quantum phase estimation (here trivial)
   qi_cx(gState, bm_psi0, bm_E_new0);
   qi_cx(gState, bm_psi1, bm_E_new1);
}

void apply_Phi_inverse(){
   // quantum phase estimation (here trivial)
   qi_cx(gState, bm_psi0, bm_E_new0);
   qi_cx(gState, bm_psi1, bm_E_new1);
}

uint draw_C(){
    if (rangen.doub()<0.5)
        return 0U;
    return 1U;
}

void apply_C(const uint &Ci){
    if(Ci==0U){
        qi_x(gState,bm_psi1);
        qi_cx(gState,bm_psi1,bm_psi0);
        qi_x(gState,bm_psi1);
    }else if(Ci==1U){
        //TODO: optimizable as SWAP gate
        qi_cx(gState,bm_psi1,bm_psi0);
        qi_cx(gState,bm_psi0,bm_psi1);
        qi_cx(gState,bm_psi1,bm_psi0);
    }else{
        throw "Error!";
    }
}

void apply_C_inverse(const uint &Ci){
    apply_C(Ci);
}

void apply_W(){
    uint mask = 124U;
    uint case1a = 80U;
    uint case1b = 100U;
    uint case2 = 96U;
//                  (1U << bm_E_old0)
//                | (1U << bm_E_old1)
//                | (1U << bm_E_new0)
//                | (1U << bm_E_new1);
    for(uint i = 0U; i < gState.size(); ++i){
        if(((i & mask) == case1a) || ((i & mask) == case1b)){
            uint j = i & ~(1U << bm_acc);
            
            apply_2x2mat(gState[i], gState[j], sqrt(1.-f1), sqrt(f1), sqrt(f1), -sqrt(1.-f1));
        }else if((i & mask) == case2){
            uint j = i & ~(1U << bm_acc);

            apply_2x2mat(gState[i], gState[j], sqrt(1.-f2), sqrt(f2), sqrt(f2), -sqrt(1.-f2));
        }else if((i >> bm_acc) & 1U){
            uint j = i & ~(1U << bm_acc);

            std::swap(gState[i],gState[j]);
        }
    }
}

void apply_W_inverse(){
    apply_W();
}

void apply_U(){
    apply_C(gCi);
    apply_Phi();
    apply_W();
}

void apply_U_inverse(){
    apply_C_inverse(gCi);
    apply_Phi_inverse();
    apply_W_inverse();
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
        for(uint i = 0U; i < gState.size(); ++i){
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


void metro_step(){
    reset_non_state_qbits();
    apply_Phi_old();

    gCi = draw_C();
    apply_U();

    measure_qbit(gState, bm_acc, c_acc);

    if (c_acc == 1U){
        cout<<"accepted"<<endl;
        vector<uint> c_E_news(2,0);
        measure_qbits(gState, {bm_E_new0, bm_E_new1}, c_E_news);
        energy_measures.push_back(c_E_news[0]+2*c_E_news[1]);
        apply_Phi_inverse();
        return;
    }
    //else

    cout<<"rejected; restoration cycle:"<<endl;
    apply_U_inverse();

    uint iters = max_reverse_attempts;
    while(iters > 0){
        apply_Phi();
        double Eold_meas, Enew_meas;
        vector<uint> c_E_olds(2,0);
        vector<uint> c_E_news(2,0);
        measure_qbits(gState, {bm_E_old0, bm_E_old1}, c_E_olds);
        Eold_meas = c_E_olds[0]+2*c_E_olds[1];
        measure_qbits(gState, {bm_E_new0, bm_E_new1}, c_E_news);
        Enew_meas = c_E_news[0]+2*c_E_news[1];
        apply_Phi_inverse();
        
        if(Eold_meas == Enew_meas){
            cout<<"  accepted restoration ("<<max_reverse_attempts-iters<<"/"<<max_reverse_attempts<<")"<<endl; 
            energy_measures.push_back(Eold_meas);
            break;
        }
            cout<<"  rejected ("<<max_reverse_attempts-iters<<"/"<<max_reverse_attempts<<")"<<endl; 
        //else
        uint c_acc_trash;
        apply_U(); 
        measure_qbit(gState, bm_acc, c_acc_trash); 
        apply_U_inverse(); 

        iters--;
    }

    if (iters == 0){
        cout<<"not converged :("<<endl;
        exit(1);
    }
}


int main(int argc, char** argv){
    if(argc < 5){
        cout<<"arguments: <beta> <eps> <metro iterations> <output file path> [--max-reverse <max reverse attempts>=20]"<<endl;
        exit(1);
    }
    beta = stod(argv[1]);
    eps = stod(argv[2]);
    metro_steps = (uint)atoi(argv[3]);
    string outfilename(argv[4]);
    max_reverse_attempts = (argc==7 && strcmp(argv[5],"--max-reverse")==0)? (uint)atoi(argv[6]) : 20U;
    
    f1 = exp(-beta*eps);
    f2 = exp(-2.*beta*eps);
    
    // Banner
    printf("\n"
		".▄▄▄  ▪  .▄▄ · ▄ •▄ ▪  ▄▄▄▄▄    .▄▄ · ▄• ▄▌ ▄▄·  ▄▄▄· \n"
		"▐▀•▀█ ██ ▐█ ▀. █▌▄▌▪██ •██      ▐█ ▀. █▪██▌▐█ ▌▪▐█ ▀█ \n" 
		"█▌·.█▌▐█·▄▀▀▀█▄▐▀▀▄·▐█· ▐█.▪    ▄▀▀▀█▄█▌▐█▌██ ▄▄▄█▀▀█ \n" 
		"▐█▪▄█·▐█▌▐█▄▪▐█▐█.█▌▐█▌ ▐█▌·    ▐█▄▪▐█▐█▄█▌▐███▌▐█ ▪▐ \n"
		"·▀▀█. ▀▀▀ ▀▀▀▀ ·▀  ▀▀▀▀ ▀▀▀      ▀▀▀▀  ▀▀▀ ·▀▀▀  ▀  ▀ \n" );

    // Initialization:
    // known eigenstate of the system: psi=0, E_old = 0
    gState[0] = 1.0; 
    energy_measures.push_back(0.0);

    for(uint s = 0U; s < metro_steps; ++s){
        metro_step();
    }


    cout<<"all fine :)\n"<<endl;

    FILE * fil = fopen(outfilename.c_str(), "w");

    fprintf(fil, "# it E\n");

    for(uint ei = 0; ei < energy_measures.size(); ++ei){
        fprintf(fil, "%d %.lg\n", ei, energy_measures[ei]);
    }
    fclose(fil);

    cout<<"\n\tSuca!\n"<<endl;

//     // test gates:
//     cout<<"TEST GATES"<<endl;
//     vector<Complex> test_state = {{0.4,-1.6},{1.2,0.7},{-0.1,0.6},{-1.3,0.4},{1.2,-1.3},{-1.2,1.7},{-3.1,0.6},{-0.3,0.2}};
//     vnormalize(test_state);
//     cout<<"initial state:"<<endl;
//     print(test_state);
//     cout<<"apply X to qbit 1 (most significant one)"<<endl;
//     qi_x(test_state, 1);
//     print(test_state);
//     cout<<"apply CX controlled by qbit 0 to qbit 1"<<endl;
//     qi_cx(test_state, 0, 1);
//     print(test_state);
//     cout<<"apply CCX controlled by qbit 1 and 2 to qbit 0"<<endl;
//     qi_mcx(test_state, {1,2}, 0);
//     print(test_state);

    return 0;
}
