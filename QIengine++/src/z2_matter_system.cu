#include "z2_matter_system.cuh"
#include <math.h>
#include "suqa.cuh"
/*  Z2 gauge theory + fermionic fields

	Open chain with 4 fermions on the sites
	and 3 z2 gauge links.
	
	Fermions -> 4 qubtis
	z2_gauge_links -> 1 qubti
	Total -> 7 Qubiti

	The structure is |f3f2f1f0G2G1G0> 
	First the gauge links and then the fermions.

*/

double m_mass;


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
	
	suqa::apply_sigmam(state, bm_z2_qferm0);
	suqa::apply_sigmam(state, bm_z2_qferm1);
	suqa::apply_z(state, bm_z2_qlink0);
}


// Apply the operator exp(-i*H_m*dt). Where H_m = \Sum_i ( -m/2(-1)^i\sigma^(z)(i)).
// If \theta_i = -m/2 * (-1)^i \delta_t in matrix form is
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
			double tmpval=state_re[i];
			state_re[i] =  cos(theta)*tmpval - sin(theta)*state_re[j];					
			state_re[j] =  cos(theta)*state_re[j] - sin(theta)*tmpval;					
			
			tmpval=state_im[i];
			state_im[i] = cos(theta)*tmpval - sin(theta)*state_im[j];					
			state_im[j] = cos(theta)*tmpval - sin(theta)*state_im[j];					
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

