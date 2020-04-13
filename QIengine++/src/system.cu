#include "system.cuh"
#include "Rand.hpp"



//TODO: make the number of "state" qubits determined at compilation time in system.cuh
double g_beta;


__global__ void initialize_state(double *state_re, double *state_im, uint len){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    while(i<len){
        state_re[i] = 0.0;
        state_im[i] = 0.0;
        i += gridDim.x*blockDim.x;
    }
    if(blockIdx.x*blockDim.x+threadIdx.x==1){
        state_re[0] = 1.0;
        state_im[0] = 0.0;
    }
}




void init_state(ComplexVec& state, uint Dim){

    if(state.size()!=Dim)
	throw std::runtime_error("ERROR: init_state() failed");
    

    initialize_state<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im,Dim);

	suqa::apply_x(state, bm_spin[1]);
	suqa::apply_h(state, bm_spin[1]);
	suqa::apply_cx(state, bm_spin[1], bm_spin[0]);
}




void exp_it_id_x_x( ComplexVec& state, const bmReg& q, uint pos_id, double phase_t){
	
	suqa::apply_pauli_TP_rotation(state, {q[(pos_id+1)%3],q[(pos_id+2)%3]}, {PAULI_X,PAULI_X}, phase_t);

}

void evolution(ComplexVec& state, const double& t, const int& n){

	for (uint iii=0; iii<3; ++iii){
		exp_it_id_x_x(state, bm_spin, iii, -t);
 	}

}


/* Measure facilities */
const uint op_bits = 3; // 2^op_bits is the number of eigenvalues for the observable
const bmReg bm_op = bm_spin; // where the measure has to be taken
const std::vector<double> op_vals = {2.0,0.0,2.0,0.0,-2.0,0.0,-2.0,0.0}; // eigvals

 
// change basis to the observable basis somewhere in the system registers
void apply_measure_rotation(ComplexVec& state){
	suqa::apply_h(state, bm_spin[0]);
	suqa::apply_h(state, bm_spin[1]);
	suqa::apply_h(state, bm_spin[2]);
	suqa::apply_cx(state, bm_spin[0], bm_spin[1]);
	suqa::apply_cx(state, bm_spin[0], bm_spin[2]);
	suqa::apply_u1(state,bm_spin[2], M_PI*0.5);
}

// inverse of the above function
void apply_measure_antirotation(ComplexVec& state){
 	apply_measure_rotation(state);
}

// map the classical measure recorded in creg_vals
// to the corresponding value of the observable;
// there is no need to change it
double get_meas_opvals(const uint& creg_vals){
    return op_vals[creg_vals];
}

// actually perform the measure
// there is no need to change it
double measure_X(ComplexVec& state, pcg& rgen){
    std::vector<uint> classics(op_bits);
    
    apply_measure_rotation(state);

    std::vector<double> rdoubs(op_bits);
    for(auto& el : rdoubs){
        el = rgen.doub();
    }
    suqa::measure_qbits(state, bm_op, classics, rdoubs);

    apply_measure_antirotation(state);

    uint meas = 0U;
    for(uint i=0; i<op_bits; ++i){
        meas |= (classics[i] << i);
    }

    return get_meas_opvals(meas);
    return 0.0;
}

/* Moves facilities */

std::vector<double> C_weigthsums = {1./3, 2./3, 1.0};




void apply_C(ComplexVec& state, const uint &Ci){
    if(Ci>2)
        throw std::runtime_error("ERROR: wrong move selection");

    suqa::apply_h(state,bm_spin[Ci]);
}


void apply_C_inverse(ComplexVec& state, const uint &Ci){
    apply_C(state, Ci);
}

std::vector<double> get_C_weigthsums(){ return C_weigthsums; }

