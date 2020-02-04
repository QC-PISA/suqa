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


// Set the initial state to |000..0> with probability 1.
void init_state(ComplexVec& state, uint Dim){

    if(state.size()!=Dim)
        throw std::runtime_error("ERROR: init_state() failed");
    
    // zeroes entries and set state to all the computational element |000...00>
    initialize_state<<<suqa::blocks,suqa::threads, 0, suqa::stream1>>>(state.data_re, state.data_im,Dim);
    cudaDeviceSynchronize();
}



