#include "system.cuh"
#include <math.h>
#include "suqa.cuh"
#include "include/Rand.hpp"
/*  Z2 gauge theory + fermionic fields

	Open chain with 4 fermions on the sites
	and 3 z2 gauge links.
	
	Fermions -> 4 qubtis
	z2_gauge_links -> 1 qubti
	Total -> 7 Qubiti

	The structure is |f3f2f1f0G2G1G0> 

*/

double m_mass;


__global__ void initialize_state(double *state_re, double *state_im, uint len){
    uint i = blockIdx.x*blockDim.x+threadIdx.x;
    while(i<len){
        state_re[i] = 0.0;
        state_im[i] = 0.0;
        i += gridDim.x*blockDim.x;
    }
    if(blockIdx.x*blockDim.x+threadIdx.x==0){
        state_re[0] = 1.0;
        state_im[0] = 0.0;
    }
}

// preparation of the state //

// Set the initial state to the projection of |000..0> on the gauge invariant subspace
// The gauge invariant state is |psi> = (1./sqrt(8))(|000>|0> + |001>|0> + |010>|0> + ... + |111>|0>)
// The |0> means that the fermions variables remain |0>.
void init_state(ComplexVec& state, uint Dim){

    if(state.size()!=Dim)
        throw std::runtime_error("ERROR: init_state() failed");
    
    // zeroes entries and set state to all the computational element |000...00>
    initialize_state<<<suqa::blocks,suqa::threads, 0, suqa::stream1>>>(state.data_re, state.data_im,Dim);
    cudaDeviceSynchronize();

	suqa::apply_h(state, bm_z2_qlink0);
	suqa::apply_h(state, bm_z2_qlink1);
	suqa::apply_h(state, bm_z2_qlink2);

}

// Apply the operator chi1_chi2_sigma_z12. This is the one used in the paper by Lamm: PRD 100, 034518 (2019).
// TODO: Generalize it.
void apply_lamm_operator(ComplexVec& state){
	
	suqa::apply_z(state, bm_z2_qlink0);
	suqa::apply_z(state, bm_z2_qferm0);
	suqa::apply_sigmam(state, bm_z2_qferm1);
	suqa::apply_sigmam(state, bm_z2_qferm0);	
}


// Apply the operator exp(-i*H_m*dt). Where H_m = \Sum_i ( -m/2(-1)^i\sigma^(z)(i)).
// If \theta_i = m/2 * (-1)^i \delta_t in matrix form is
// cos(theta)1 + i*sen(theta)sigma^(z)(i)
__global__ 
void kernel_apply_mass_evolution(double *state_re, double *state_im, uint len, uint q, double theta){
	uint i = blockIdx.x*blockDim.x+threadIdx.x;
	uint glob_mask = 0U;

	glob_mask |= (1U << q);
	while(i<len){
		if((i & glob_mask) == glob_mask){
			uint j = i & ~(1U << q); // j has 0 in the site 0 qubit
			double tmpval_re=state_re[i];
			double tmpval_im=state_im[i];
			state_re[i] =  cos(theta)*tmpval_re + sin(theta)*tmpval_im;					
			state_im[i] = -sin(theta)*tmpval_re + cos(theta)*tmpval_im;					

			tmpval_re=state_re[j];
			tmpval_im=state_im[j];			
			state_re[j] =  cos(theta)*tmpval_re - sin(theta)*tmpval_im;					
			state_im[j] =  sin(theta)*tmpval_re + cos(theta)*tmpval_im;
		}
		i+=gridDim.x*blockDim.x;
	}
}
 
void apply_mass_evolution(ComplexVec& state, uint q, double theta){
	kernel_apply_mass_evolution<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, theta);
}
 
void apply_mass_evolution(ComplexVec& state, const bmReg& qs, double theta){
	for(const auto&q:qs)
		kernel_apply_mass_evolution<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, theta);
}

// Apply the operator exp(-i*H_kg*dt). Where H_kg \Sum_i ( \sigma^(x)(i, i+1)). It is applied on the links variables.
// The matrix form is 
// cos(dt)1 - i*sen(dt)sigma^(x)(i, i+1)
__global__ 
void kernel_apply_gauge_link_evolution(double *state_re, double *state_im, uint len, uint q, double theta){
	uint i = blockIdx.x*blockDim.x+threadIdx.x;
	uint glob_mask = 0U;

	glob_mask |= (1U << q);
	while(i<len){
		if((i & glob_mask) == glob_mask){
			uint j = i & ~(1U << q); // j has 0 in the site 0 qubit
			double tmpval_re=state_re[i];
			double tmpval_im=state_im[i];
			state_re[i] = cos(theta)*tmpval_re - sin(theta)*state_im[j];
			state_im[i] = tmpval_im*cos(theta) + state_re[j]*sin(theta); 
			
			
			state_re[j] = cos(theta)*state_re[j] - sin(theta)*tmpval_im;
			state_im[j] = state_im[j]*cos(theta) + tmpval_re*sin(theta); 
		}
		i+=gridDim.x*blockDim.x;
	}
}
 
void apply_gauge_link_evolution(ComplexVec& state, uint q, double theta){
	kernel_apply_gauge_link_evolution<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, theta);
}
 
void apply_gauge_link_evolution(ComplexVec& state, const bmReg& qs, double theta){
	for(const auto&q:qs)
		kernel_apply_gauge_link_evolution<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, theta);
}

// Apply the operator exp(-i*H_hx*dt). Where H_hx \Sum_i ( (1/4)*(-1)^i*\sigma^{z}(i,i+1)\sigma^(x)(i)\sigma^{x}(i+1)). It is applied on 
//both on the links variables and on the sites.
// The inputs are qlink, the link at i,i+1 -- qferm_m the fermion at site i -- and qferm_p the fermion at site i+1
// The matrix form is (theta = -1/4*(-1)^idt):
//  cos(theta)1 + isen(theta)sigma^{z}(i,i+1) 
//  cos(theta)1 + isen(theta)sigma^{x}(i)
//  cos(theta)1 + isen(theta)sigma^{x}(i+1)
__global__ 
void kernel_apply_hopping_evolution_x(double *state_re, double *state_im, uint len, uint qlink, uint qferm_m, uint qferm_p, double theta){
	uint i = blockIdx.x*blockDim.x+threadIdx.x;
	
	uint mask = 0U;
	
	mask |= (1U << qlink);
	mask |= (1U << qferm_m);
	mask |= (1U << qferm_p);

	double tmpval;

	while(i<len){
		if((i & mask) == 0U){

			//state[i] = i_0
			uint i_1 = i | (1U << qlink);
			uint i_2 = i | (1U << qferm_m);
			uint i_3 = i_1 | i_2;
			uint i_4 = i | (1U << qferm_p);
			uint i_5 = i_4 | i_1;
			uint i_6 = i_4 | i_2;
			uint i_7 = i_4 | i_3;
	
			//i_0 and i_6 couple
			tmpval=state_re[i];
			state_re[i] = tmpval*cos(theta) - state_im[i_6]*sin(theta);
			state_im[i_6] = state_im[i_6]*cos(theta) + tmpval*sin(theta);

			tmpval=state_im[i];
			state_im[i] = tmpval*cos(theta) + state_re[i_6]*sin(theta);
			state_re[i_6] = state_re[i_6]*cos(theta) - tmpval*sin(theta);

			
			//i_1 and i_7 couple
			tmpval=state_re[i_1];
			state_re[i_1] = tmpval*cos(theta) + state_im[i_7]*sin(theta);
			state_im[i_7] = state_im[i_7]*cos(theta) - tmpval*sin(theta);

			tmpval=state_im[i_1];
			state_im[i_1] = tmpval*cos(theta) - state_re[i_7]*sin(theta);
			state_re[i_7] = state_re[i_7]*cos(theta) + tmpval*sin(theta);


			//i_2 and i_4 couple
			tmpval=state_re[i_2];
			state_re[i_2] = tmpval*cos(theta) - state_im[i_4]*sin(theta);
			state_im[i_4] = state_im[i_4]*cos(theta) + tmpval*sin(theta);

			tmpval=state_im[i_2];
			state_im[i_2] = tmpval*cos(theta) + state_re[i_4]*sin(theta);
			state_re[i_4] = state_re[i_4]*cos(theta) - tmpval*sin(theta);


			//i_3 and i_5 couple
			tmpval=state_re[i_3];
			state_re[i_3] = tmpval*cos(theta) + state_im[i_5]*sin(theta);
			state_im[i_5] = state_im[i_5]*cos(theta) - tmpval*sin(theta);

			tmpval=state_im[i_3];
			state_im[i_3] = tmpval*cos(theta) - state_re[i_5]*sin(theta);
			state_re[i_5] = state_re[i_5]*cos(theta) + tmpval*sin(theta);
		}
		i+=gridDim.x*blockDim.x;
	}
}
 
void apply_hopping_evolution_x(ComplexVec& state, uint qlink, uint qferm_m, uint qferm_p, double theta){
	kernel_apply_hopping_evolution_x<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), qlink, qferm_m, qferm_p, theta);
}
 
// Apply the operator exp(-i*H_hy*dt). Where H_hy \Sum_i ( (1/4)*(-1)^i*\sigma^{z}(i,i+1)\sigma^(y)(i)\sigma^{y}(i+1)). It is applied on 
//both on the links variables and on the sites.
// The inputs are qlink, the link at i,i+1 -- qferm_m the fermion at site i -- and qferm_p the fermion at site i+1
// The matrix form is (theta = -1/4*(-1)^idt):
//  cos(theta)1 + isen(theta)sigma^{z}(i,i+1) 
//  cos(theta)1 + isen(theta)sigma^{y}(i)
//  cos(theta)1 + isen(theta)sigma^{y}(i+1)
__global__ 
void kernel_apply_hopping_evolution_y(double *state_re, double *state_im, uint len, uint qlink, uint qferm_m, uint qferm_p, double theta){
	uint i = blockIdx.x*blockDim.x+threadIdx.x;
	
	uint mask = 0U;
	
	mask |= (1U << qlink);
	mask |= (1U << qferm_m);
	mask |= (1U << qferm_p);

	double tmpval;

	while(i<len){
		if((i & mask) == 0U){

			//state[i] = i_0
			uint i_1 = i | (1U << qlink);
			uint i_2 = i | (1U << qferm_m);
			uint i_3 = i_1 | i_2;
			uint i_4 = i | (1U << qferm_p);
			uint i_5 = i_4 | i_1;
			uint i_6 = i_4 | i_2;
			uint i_7 = i_4 | i_3;
	
			//i_0 and i_6 couple
			tmpval=state_re[i];
			state_re[i] = tmpval*cos(theta) + state_im[i_6]*sin(theta);
			state_im[i_6] = state_im[i_6]*cos(theta) - tmpval*sin(theta);

			tmpval=state_im[i];
			state_im[i] = tmpval*cos(theta) - state_re[i_6]*sin(theta);
			state_re[i_6] = state_re[i_6]*cos(theta) + tmpval*sin(theta);

			
			//i_1 and i_7 couple
			tmpval=state_re[i_1];
			state_re[i_1] = tmpval*cos(theta) - state_im[i_7]*sin(theta);
			state_im[i_7] = state_im[i_7]*cos(theta) + tmpval*sin(theta);

			tmpval=state_im[i_1];
			state_im[i_1] = tmpval*cos(theta) + state_re[i_7]*sin(theta);
			state_re[i_7] = state_re[i_7]*cos(theta) - tmpval*sin(theta);


			//i_2 and i_4 couple
			tmpval=state_re[i_2];
			state_re[i_2] = tmpval*cos(theta) - state_im[i_4]*sin(theta);
			state_im[i_4] = state_im[i_4]*cos(theta) + tmpval*sin(theta);

			tmpval=state_im[i_2];
			state_im[i_2] = tmpval*cos(theta) + state_re[i_4]*sin(theta);
			state_re[i_4] = state_re[i_4]*cos(theta) - tmpval*sin(theta);


			//i_3 and i_5 couple
			tmpval=state_re[i_3];
			state_re[i_3] = tmpval*cos(theta) + state_im[i_5]*sin(theta);
			state_im[i_5] = state_im[i_5]*cos(theta) - tmpval*sin(theta);

			tmpval=state_im[i_3];
			state_im[i_3] = tmpval*cos(theta) - state_re[i_5]*sin(theta);
			state_re[i_5] = state_re[i_5]*cos(theta) + tmpval*sin(theta);
		}
		i+=gridDim.x*blockDim.x;
	}
}
 
void apply_hopping_evolution_y(ComplexVec& state, uint qlink, uint qferm_m, uint qferm_p, double theta){
	kernel_apply_hopping_evolution_y<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), qlink, qferm_m, qferm_p, theta);
}

void evolution(ComplexVec& state, const double& t, const int& n){
	const double dt = t/(double)n;

	const double mass_coef = -dt*m_mass*0.5; // remember the parity of the site
	const double gauge_coef = -dt;
	const double hopping_theta = -dt*0.25; // remember the parity of the site

	for (uint ti=0; ti<(uint)n; ++ti){
		DEBUG_CALL(printf("Initial state()\n"));
		DEBUG_READ_STATE(state);
	//	apply_hopping_evolution_y(state, bm_z2_qlink0[0], bm_z2_qferm0[0], bm_z2_qferm1[0], -hopping_theta);
		
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qlink0[0], bm_z2_qferm0[0], bm_z2_qferm1[0]}, {PAULI_Z,PAULI_Y,PAULI_Y}, -hopping_theta);
		DEBUG_CALL(printf("After hopping evolution y site 0()\n"));
		DEBUG_READ_STATE(state);
		
//		apply_hopping_evolution_y(state, bm_z2_qlink1[0], bm_z2_qferm1[0], bm_z2_qferm2[0], hopping_theta);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qlink1[0], bm_z2_qferm1[0], bm_z2_qferm2[0]}, {PAULI_Z,PAULI_Y,PAULI_Y}, +hopping_theta);
		DEBUG_CALL(printf("After hopping evolution y site 1 ()\n"));
		DEBUG_READ_STATE(state);

//		apply_hopping_evolution_y(state, bm_z2_qlink2[0], bm_z2_qferm2[0], bm_z2_qferm3[0], -hopping_theta);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qlink2[0], bm_z2_qferm2[0], bm_z2_qferm3[0]}, {PAULI_Z,PAULI_Y,PAULI_Y}, -hopping_theta);
		DEBUG_CALL(printf("After hopping evolution y site 2 ()\n"));
		DEBUG_READ_STATE(state);

//		apply_hopping_evolution_x(state, bm_z2_qlink0[0], bm_z2_qferm0[0], bm_z2_qferm1[0], -hopping_theta);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qlink0[0], bm_z2_qferm0[0], bm_z2_qferm1[0]}, {PAULI_Z,PAULI_X,PAULI_X}, -hopping_theta);
		DEBUG_CALL(printf("After hopping evolution x site 0()\n"));
		DEBUG_READ_STATE(state);
		
//		apply_hopping_evolution_x(state, bm_z2_qlink1[0], bm_z2_qferm1[0], bm_z2_qferm2[0], hopping_theta);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qlink1[0], bm_z2_qferm1[0], bm_z2_qferm2[0]}, {PAULI_Z,PAULI_X,PAULI_X}, +hopping_theta);
		DEBUG_CALL(printf("After hopping evolution x site 1 ()\n"));
		DEBUG_READ_STATE(state);

//		apply_hopping_evolution_x(state, bm_z2_qlink2[0], bm_z2_qferm2[0], bm_z2_qferm3[0], -hopping_theta);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qlink2[0], bm_z2_qferm2[0], bm_z2_qferm3[0]}, {PAULI_Z,PAULI_X,PAULI_X}, -hopping_theta);
		DEBUG_CALL(printf("After hopping evolution x site 2 ()\n"));
		DEBUG_READ_STATE(state);

//gauge link part


//		apply_gauge_link_evolution(state, bm_z2_qlink0, gauge_coef);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qlink0[0]}, {PAULI_X}, gauge_coef);
		DEBUG_CALL(printf("After gauge evolution  link0 ()\n"));
		DEBUG_READ_STATE(state);

//		apply_gauge_link_evolution(state, bm_z2_qlink1, gauge_coef);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qlink1[0]}, {PAULI_X}, gauge_coef);
		DEBUG_CALL(printf("After gauge evolution  link1 ()\n"));
		DEBUG_READ_STATE(state);

//		apply_gauge_link_evolution(state, bm_z2_qlink2, gauge_coef);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qlink2[0]}, {PAULI_X}, gauge_coef);
		DEBUG_CALL(printf("After gauge evolution  link2 ()\n"));
		DEBUG_READ_STATE(state);

// fermion mass part

//		apply_mass_evolution(state, bm_z2_qferm0, -mass_coef);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qferm0[0]}, {PAULI_Z}, -mass_coef);
		DEBUG_CALL(printf("After mass evolution site 0 ()\n"));
		DEBUG_READ_STATE(state);

//		apply_mass_evolution(state, bm_z2_qferm1, mass_coef);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qferm1[0]}, {PAULI_Z}, +mass_coef);
		DEBUG_CALL(printf("After mass evolution site 1 ()\n"));
		DEBUG_READ_STATE(state);

//		apply_mass_evolution(state, bm_z2_qferm2, -mass_coef);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qferm2[0]}, {PAULI_Z}, -mass_coef);
		DEBUG_CALL(printf("After mass evolution site 2 ()\n"));
		DEBUG_READ_STATE(state);

//		apply_mass_evolution(state, bm_z2_qferm3, mass_coef);
		suqa::apply_pauli_TP_rotation(state, {bm_z2_qferm3[0]}, {PAULI_Z}, mass_coef);
		DEBUG_CALL(printf("After mass evolution site 3 ()\n"));
		DEBUG_READ_STATE(state);
	}
}


/******* QMS and Measures *******/

const uint op_bits = 3;
const bmReg bm_op = bm_z2_qferm0; // where the measure has to be taken

std::vector<double> C_weigthsums = {1./12, 2./12, 3./12, 4./12, 5./12, 6./12, 
									7./12, 8./12, 9./12, 10./12, 11./12, 1.};

std::vector<bmReg> link = {bm_z2_qlink0, bm_z2_qlink1, bm_z2_qlink2};
std::vector<bmReg> ferm = {bm_z2_qferm0, bm_z2_qferm1, bm_z2_qferm2, bm_z2_qferm3};

double measure_X(ComplexVec& state, pcg& rgen){
//	std::vector<uint> classics(op_bits);
//
//	std::vector<double> rdoubs(op_bits);
//	for(auto& el :rdoubs){
//		el = rgen.doub();
//	}
//
//	suqa::measure_qbits(state, bm_op, classics, rdoubs);
//
//	uint meas = 0U;
//	for(uint i=0; i<op_bits; ++i){
//		meas |= (classics[i] << i);
//	}	
//
	return 0;
}

void apply_C_inverse(ComplexVec& state, const uint& Ci){
	int link_index = Ci%3;
//	int ferm_index = Ci%4;
	int ferm_index = link_index+1;

//	suqa::apply_u1(state, ferm[ferm_index][0], -M_PI);
	suqa::apply_x(state, ferm[ferm_index]);
	suqa::apply_x(state, ferm[ferm_index-1]);		
//	suqa::apply_x(state, link[link_index]);	
	suqa::apply_z(state, link[link_index]);	
	suqa::apply_z(state, ferm[ferm_index-1]);		
//	suqa::apply_u1(state, ferm[ferm_index][0], -M_PI);
//	for(int i=0;i<ferm_index;i++){
//		suqa::apply_z(state, ferm[i]);
//	}	

}

void apply_C(ComplexVec& state, const uint& Ci){
	int link_index = Ci%3;
//	int ferm_index = Ci%4;
	int ferm_index = link_index+1;

//	suqa::apply_u1(state, ferm[ferm_index][0], -M_PI);
	suqa::apply_x(state, ferm[ferm_index]);
	suqa::apply_z(state, ferm[ferm_index-1]);		
	suqa::apply_x(state, ferm[ferm_index-1]);		
	suqa::apply_z(state, link[link_index]);	
//	suqa::apply_u1(state, ferm[ferm_index][0], -M_PI);
//	for(int i=0;i<ferm_index;i++){
//		suqa::apply_z(state, ferm[i]);
//	}	

//	suqa::apply_x(state, link[link_index]);	
}

std::vector<double> get_C_weigthsums() { return C_weigthsums;} 





