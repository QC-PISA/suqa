#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

typedef complex<double> Complex;
const Complex iu(0, 1);

/* Hamiltonian
 *
 * H = eps diag(0, 1, 2)
 *
 */


// simulation parameters
const double beta = 1.0;
const double eps = 1.0;
const double th1 = 2*asin(exp(-beta*eps));
const double th2 = 2*asin(exp(-2*beta*eps));
const uint nqubits = 8;
const uint Dim = (uint)pow(2.0, nqubits);

// simulation hyperparameters
const uint max_reverse_attempts = 20;
const uint metro_steps = 100;

// Global state of the system.
// Ordering (less to most significant)
// psi[0], psi[1], E_old[0], E_old[1], E_new[0], E_new[1], acc, qaux[0]
vector<Complex> gState(Dim,0.0);

vector<double> energy_measures;


// Utilities

// bit masks
enum bm_idxs {  bm_psi0, 
                bm_psi1,
                bm_E_old0,
                bm_E_old1,
                bm_E_new0,
                bm_E_new1,
                bm_acc,
                bm_qaux0};
// const vector<uint> bm_list = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};


// vector<uint> get_binary(const uint& i){
//     vector<uint> ret(nqubits);
//     for(uint j = 0; j < nqubits; ++j){
//         ret[j] = i & (1U << j);
//     }
//     return ret;
// }
// 
// uint get_decimal(const vector<uint>& b){
//     uint ret = 0;
//     for(uint j = 0; j < nqubits; ++j)
//         ret += b[j] << j;
//     
//     return ret;
// }

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

void apply_Phi(){
   // quantum phase estimation (here trivial)
   qi_cx(gState, bm_psi0, bm_E_new0);
   qi_cx(gState, bm_psi1, bm_E_new1);
}


void metro_step(){
    reset_non_state_qbits();
    apply_Phi();

}



int main(){
    
    // Initialization:
    // known eigenstate of the system: psi=0, E_old = 0
    gState[0] = 1.0; 
    energy_measures.push_back(0.0);

//    for(uint s = 0U; s < metro_steps; ++s){
//        metro_step();
//    }

//TODO: continue

    // test gates:
    cout<<"TEST GATES"<<endl;
    vector<Complex> test_state = {{0.4,-1.6},{1.2,0.7},{-0.1,0.6},{-1.3,0.4},{1.2,-1.3},{-1.2,1.7},{-3.1,0.6},{-0.3,0.2}};
    vnormalize(test_state);
    cout<<"initial state:"<<endl;
    print(test_state);
    cout<<"apply X to qbit 1 (most significant one)"<<endl;
    qi_x(test_state, 1);
    print(test_state);
    cout<<"apply CX controlled by qbit 0 to qbit 1"<<endl;
    qi_cx(test_state, 0, 1);
    print(test_state);
    cout<<"apply CCX controlled by qbit 1 and 2 to qbit 0"<<endl;
    qi_mcx(test_state, {1,2}, 0);
    print(test_state);

    return 0;
}
