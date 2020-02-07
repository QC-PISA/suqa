#include "z2_matter_system.cuh"

/*  Z2 gauge theory + fermionic fields

	Open chain with 4 fermions on the sites
	and 3 z2 gauge links.
	
	Fermions -> 4 qubtis
	z2_gauge_links -> 1 qubti
	Total -> 7 Qubiti

	The structure is |G1G2G3f1f2f3f4> 
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

// Apply the operator chi1_chi2_sigma_z12.




 


