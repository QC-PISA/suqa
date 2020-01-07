#include "suqa.cuh"

/*XXX: everything is optimizable
 *possible strategies:
 *
 *  * sparse data structure: worse in the worst case, but allows bigger states
 *    and potential speedup (depends on the algorithm)
 *    - caching: only useful with sparse data structures, due to bad scaling
 *      in storage consumption.
 *      Allocating it in constant memory would be pretty (only 64KB allowed, though)
 * 
 *  * grey encoding + shared memory: to improve coalescing
 *
 *  * exploit cpu: instead of caching for example, to precompute masks (e.g. in mcx) 
 */


//TODO: optimize reduce
__global__ void kernel_suqa_vnorm(double *dev_partial_ret, Complex *v, uint len){
    extern __shared__ double local_ret[];
    uint tid =  threadIdx.x;
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

    local_ret[tid] = 0.0; // + norm(v[i+blockDim.x]);
    while(i<len){
        local_ret[tid] += norm(v[i]);
        i += gridDim.x*blockDim.x;
    }
    __syncthreads();

    for(uint s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            local_ret[tid] += local_ret[tid+s];
        }
        __syncthreads();
    }

    if(tid==0) dev_partial_ret[blockIdx.x] = local_ret[0];
}

double suqa::vnorm(const ComplexVec& v){
    double ret = 0.0;
#if defined(CUDA_HOST)
    for(uint i=0; i<v.size(); ++i){
        ret += norm(v[i]);
    }
#else // CUDA defined
    double *host_partial_ret = new double[suqa::blocks];
    double *dev_partial_ret;
    cudaMalloc((void**)&dev_partial_ret, suqa::blocks*sizeof(double));  
    kernel_suqa_vnorm<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, v.data, v.size());

    cudaMemcpy(host_partial_ret,dev_partial_ret,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dev_partial_ret); 
    
    for(uint bid=0; bid<suqa::blocks; ++bid){
        ret += host_partial_ret[bid]; 
    } 
    delete [] host_partial_ret;
#endif
    return sqrt(ret);
}

__global__ void kernel_suqa_vnormalize_by(Complex *v, uint len, double value){
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;
    while(i < len){
        v[i]/=value;
        i += gridDim.x*blockDim.x;
    }
}

void suqa::vnormalize(ComplexVec& v){
    double vec_norm = suqa::vnorm(v);
#ifndef NDEBUG
    std::cout<<"vec_norm = "<<vec_norm<<std::endl;
#endif
#if defined(CUDA_HOST)
    for(uint i=0; i<v.size(); ++i){
        v[i]/=vec_norm;
    }
#else // CUDA defined
    kernel_suqa_vnormalize_by<<<suqa::blocks,suqa::threads>>>(v.data, v.size(),vec_norm);
#endif
}



//  X GATE


__global__ 
void kernel_suqa_x(Complex *const state, uint len, uint q){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if(i & (1U << q)){
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            suqa::swap_cmpx(&state[i],&state[j]);
        }
        i+=gridDim.x*blockDim.x;
    }
}


void suqa::apply_x(ComplexVec& state, uint q){
#if defined(CUDA_HOST)
    for(uint i=0U; i<state.size(); ++i){
        if(i & (1U << q)){
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            suqa::swap_cmpx(&state[i],&state[j]);
        }
    }
#else // CUDA defined
    kernel_suqa_x<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), q);
#endif
}  

// no way...
//__global__ 
//void kernel_suqa_mx(Complex *const state, uint len, uint msq, uint mask){
//    // msq here stands for most significant qubit
//    uint i = blockDim.x*blockIdx.x + threadIdx.x;
//    while(i<len){
//        if(i & (1U << msq)){
//            uint j = i & ~(1U << msq); // j has 0 on q-th digit
//            swap_cmpx(&state[i],&state[j]);
//        }
//        i+=gridDim.x*blockDim.x;
//    }
//}


void suqa::apply_x(ComplexVec& state, const std::vector<uint>& qs){
#if defined(CUDA_HOST)
    for(const auto& q : qs)
        suqa::apply_x(state, q);
#else // CUDA defined
    for(const auto& q : qs)
        kernel_suqa_x<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), q);
#endif
}  
//void suqa::qi_x(ComplexVec& state, const vector<uint>& qs){
//    for(const auto& q : qs)
//        qi_x(state, q);
//}  

//  HADAMARD GATE

__global__ 
void kernel_suqa_h(Complex *const state, uint len, uint q){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    uint i_0 = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i_0<len){
        if((i_0 & (1U << q)) == 0U){
            const uint i_1 = i_0 | (1U << q);
            Complex a_0 = state[i_0];
            Complex a_1 = state[i_1];
            
            state[i_0].x = TWOSQINV*(a_0.x+a_1.x);
            state[i_0].y = TWOSQINV*(a_0.y+a_1.y);
            state[i_1].x = TWOSQINV*(a_0.x-a_1.x);
            state[i_1].y = TWOSQINV*(a_0.y-a_1.y);
//            state[i_0] = cuCmul(TWOSQINV_CMPX,cuCadd(a_0,a_1));
//            state[i_1] = cuCmul(TWOSQINV_CMPX,cuCsub(a_0,a_1));
        }
        i_0+=gridDim.x*blockDim.x;
    }
}


void suqa::apply_h(ComplexVec& state, uint q){
#if defined(CUDA_HOST)
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
    for(uint i_0=0; i_0<state.size(); ++i_0){
        if((i_0 & (1U << q)) == 0U){
            const int i_1 = i_0 | (1U << q);
            Complex a_0 = state[i_0];
            Complex a_1 = state[i_1];
            
            state[i_0].x = TWOSQINV*(a_0.x+a_1.x);
            state[i_0].y = TWOSQINV*(a_0.y+a_1.y);
            state[i_1].x = TWOSQINV*(a_0.x-a_1.x);
            state[i_1].y = TWOSQINV*(a_0.y-a_1.y);
        }
    }
#else // CUDA defined
    kernel_suqa_h<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), q);
#endif
}  


void suqa::apply_h(ComplexVec& state, const std::vector<uint>& qs){
#if defined(CUDA_HOST)
    for(const auto& q : qs){
        suqa::apply_h(state, q);
    }
#else // CUDA defined
    for(const auto& q : qs){
        kernel_suqa_h<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), q);
    }
#endif
}  

//  CONTROLLED-NOT GATE

// old single qbit version
//__global__ 
//void kernel_suqa_cx(Complex *const state, uint len, uint q_control, uint q_target, uint mask_qs, mask){
//    int i = blockDim.x*blockIdx.x + threadIdx.x;    
//    while(i<len){
//        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
//            uint j = i & ~(1U << q_target);
//            swap_cmpx(&state[i],&state[j]);
//        }
//        i+=gridDim.x*blockDim.x;
//    }
//}

__global__ 
void kernel_suqa_mcx(Complex *const state, uint len, uint control_mask, uint mask_qs, uint q_target){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i & control_mask) == mask_qs){
            uint j = i & ~(1U << q_target);
            suqa::swap_cmpx(&state[i],&state[j]);
        }
        i+=gridDim.x*blockDim.x;
    }
}


//void suqa::apply_cx(ComplexVec& state, uint q_control, uint q_target){
////#if defined(CUDA_HOST)
////    for(uint i=0; i<state.size(); ++i){
////        if((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
////            uint j = i & ~(1U << q_target);
////            swap_cmpx(&state[i],&state[j]);
////        }
////    }
////#else // CUDA defined
////    kernel_suqa_cx<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), q_control, q_target);
////#endif
//    suqa::apply_cx(state, q_control, 1, q_target);
//}  

void suqa::apply_cx(ComplexVec& state, const uint& q_control, const uint& q_target, const uint& q_mask){
    uint mask_qs = 1U << q_target;
    uint mask = mask_qs | (1U << q_control);
    if(q_mask) mask_qs |= (1U << q_control);
#if defined(CUDA_HOST)
    for(uint i = 0U; i < state.size(); ++i){
        if((i & mask) == mask_qs){
            uint j = i & ~(1U << q_target);
            suqa::swap_cmpx(&state[i],&state[j]);
        }
    }
#else // CUDA defined
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), mask, mask_qs, q_target);
#endif
}  

void suqa::apply_mcx(ComplexVec& state, const std::vector<uint>& q_controls, const uint& q_target){
    uint mask = 1U << q_target;
    for(const auto& q : q_controls)
        mask |= 1U << q;
#if defined(CUDA_HOST)
    for(uint i = 0U; i < state.size(); ++i){
        if((i & mask) == mask){
            uint j = i & ~(1U << q_target);
            suqa::swap_cmpx(&state[i],&state[j]);
        }
    }
#else // CUDA defined
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), mask, mask, q_target);
#endif
}  


void suqa::apply_mcx(ComplexVec& state, const std::vector<uint>& q_controls, const std::vector<uint>& q_mask, const uint& q_target){
    uint mask = 1U << q_target;
    for(const auto& q : q_controls)
        mask |= 1U << q;
    uint mask_qs = 1U << q_target;
    for(uint k = 0U; k < q_controls.size(); ++k){
        if(q_mask[k]) mask_qs |= 1U << q_controls[k];
    }

#if defined(CUDA_HOST)
    for(uint i = 0U; i < state.size(); ++i){
        if((i & mask) == mask_qs){
            uint j = i & ~(1U << q_target);
            suqa::swap_cmpx(&state[i],&state[j]);
        }
    }
#else // CUDA defined
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), mask, mask_qs, q_target);
#endif
}  


__global__ 
void kernel_suqa_swap(Complex *const state, uint len, uint mask, uint mask_q, uint q2){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i & mask) == mask_q){
            uint j = (i & ~mask_q) | (1U << q2);
            suqa::swap_cmpx(&state[i],&state[j]);
        }
        i+=gridDim.x*blockDim.x;
    }
}

void suqa::apply_swap(ComplexVec& state, const uint& q1, const uint& q2){
    // swap gate: 00->00, 01->10, 10->01, 11->11
    // equivalent to cx(q1,q2)->cx(q2,q1)->cx(q1,q2)
    uint mask_q = (1U << q1);
    uint mask = mask_q | (1U << q2);
#if defined(CUDA_HOST)
    for(uint i = 0U; i < state.size(); ++i){
        if((i & mask) == mask_q){
            uint j = (i & ~mask_q) | (1U << q2);
            suqa::swap_cmpx(&state[i],&state[j]);
        }
    }
#else // CUDA defined
    kernel_suqa_swap<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), mask, mask_q, q2);
#endif
}

//__global__ 
//void kernel_suqa_mcx(Complex *const state, uint len, uint mask, uint q_target){
//    int i = blockDim.x*blockIdx.x + threadIdx.x;    
//    while(i<len){
//        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
//            uint j = i & ~(1U << q_target);
//            swap_cmpx(&state[i],&state[j]);
//        }
//        i+=gridDim.x*blockDim.x;
//    }
//}
//
//void suqa::apply_cx(ComplexVec& state, uint q_control, uint q_target){
////    uint mask = 1U << q_target;
////    for(const auto& q : q_controls)
////        mask |= 1U << q;
////
//#if defined(CUDA_HOST)
//    for(uint i=0; i<state.size(); ++i){
//        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
//            uint j = i & ~(1U << q_target);
//            swap_cmpx(&state[i],&state[j]);
//        }
//    }
//#else // CUDA defined
//    kernel_suqa_cx<<<suqa::blocks,suqa::threads>>>(state.data, state.size(), q_control, q_target);
//#endif
//}  

//void suqa::qi_x(ComplexVec& state, const vector<uint>& qs){
//    for(const auto& q : qs)
//        qi_x(state, q);
//}  
//
//void suqa::qi_h(ComplexVec& state, const uint& q){
//	for(uint i_0 = 0U; i_0 < state.size(); ++i_0){
//        if((i_0 & (1U << q)) == 0U){
//            uint i_1 = i_0 | (1U << q);
//            Complex a_0 = state[i_0];
//            Complex a_1 = state[i_1];
//            state[i_0] = TWOSQINV*(a_0+a_1);
//            state[i_1] = TWOSQINV*(a_0-a_1);
//        }
//    }
//}  
//
//void suqa::qi_h(ComplexVec& state, const vector<uint>& qs){
//    for(const auto& q : qs)
//        qi_h(state, q);
//}  
//
//
//void suqa::qi_cx(ComplexVec& state, const uint& q_control, const uint& q_target){
//    for(uint i = 0U; i < state.size(); ++i){
//        // for the swap, not only q_target:1 but also q_control:1
//        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
//            uint j = i & ~(1U << q_target);
//            std::swap(state[i],state[j]);
//        }
//    }
//}  
//  
//
//void suqa::qi_cx(ComplexVec& state, const uint& q_control, const uint& q_mask, const uint& q_target){
//    uint mask_qs = 1U << q_target;
//    uint mask = mask_qs | (1U << q_control);
//    if(q_mask) mask_qs |= (1U << q_control);
//    for(uint i = 0U; i < state.size(); ++i){
//        if((i & mask) == mask_qs){
//            uint j = i & ~(1U << q_target);
//            std::swap(state[i],state[j]);
//        }
//    }
//}  
//
//void suqa::qi_mcx(ComplexVec& state, const vector<uint>& q_controls, const uint& q_target){
//    uint mask = 1U << q_target;
//    for(const auto& q : q_controls)
//        mask |= 1U << q;
//
//    for(uint i = 0U; i < state.size(); ++i){
//        if((i & mask) == mask){
//            uint j = i & ~(1U << q_target);
//            std::swap(state[i],state[j]);
//        }
//    }
//}  
//
//
//void suqa::qi_mcx(ComplexVec& state, const vector<uint>& q_controls, const vector<uint>& q_mask, const uint& q_target){
//    uint mask = 1U << q_target;
//    for(const auto& q : q_controls)
//        mask |= 1U << q;
//    uint mask_qs = 1U << q_target;
//    for(uint k = 0U; k < q_controls.size(); ++k){
//        if(q_mask[k]) mask_qs |= 1U << q_controls[k];
//    }
//    for(uint i = 0U; i < state.size(); ++i){
//        if((i & mask) == mask_qs){
//            uint j = i & ~(1U << q_target);
//            std::swap(state[i],state[j]);
//        }
//    }
//}  
//void suqa::qi_swap(ComplexVec& state, const uint& q1, const uint& q2){
//        // swap gate: 00->00, 01->10, 10->01, 11->11
//        // equivalent to cx(q1,q2)->cx(q2,q1)->cx(q1,q2)
//        uint mask_q = (1U << q1);
//        uint mask = mask_q | (1U << q2);
//        for(uint i = 0U; i < state.size(); ++i){
//            if((i & mask) == mask_q){
//                uint j = (i & ~(1U << q1)) | (1U << q2);
//                std::swap(state[i],state[j]);
//            }
//        }
//}

// RESET = measure + classical cx

// sets amplitudes with value <val> in qubit <q> to zero
// !! it leaves the state unnormalized !!
__global__ void kernel_suqa_set_ampl_to_zero(Complex *state, uint len, uint q, uint val){
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;
    while(i<len){
        if(((i >> q) & 1U) == val){
            state[i].x = 0.0;
            state[i].y = 0.0;
        }
        i += gridDim.x*blockDim.x;
    }
}


void set_ampl_to_zero(ComplexVec& state, const uint& q, const uint& val){
#if defined(CUDA_HOST)
    for(uint i = 0U; i < state.size(); ++i){
        if(((i >> q) & 1U) == val){
            state[i].x = 0.0;
            state[i].y = 0.0;
        }
    }
#else // CUDA defined
    kernel_suqa_set_ampl_to_zero<<<suqa::blocks, suqa::threads>>>(state.data, state.size(), q, val);
#endif
}

//TODO: optimize reduce
__global__ void kernel_suqa_prob1(double *dev_partial_ret, Complex *v, uint len, uint q, double rdoub){
    extern __shared__ double local_ret[];
    uint tid =  threadIdx.x;
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

    local_ret[tid] = 0.0; // + norm(v[i+blockDim.x]);
    while(i<len){
        if(i & (1U << q)){
            local_ret[tid] += norm(v[i]);
        }
        i += gridDim.x*blockDim.x;
    }
    __syncthreads();

    for(uint s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            local_ret[tid] += local_ret[tid+s];
        }
        __syncthreads();
    }

    if(tid==0) dev_partial_ret[blockIdx.x] = local_ret[0];
}

void suqa::measure_qbit(ComplexVec& state, uint q, uint& c, double rdoub){
    double prob1 = 0.0;
    c=0U;
#if defined(CUDA_HOST)
    for(uint i = 0U; i < state.size() && prob1<rdoub; ++i){
        if(i & (1U << q)){
            prob1+=norm(state[i]); 
        }
    }
#else // CUDA defined
    double *host_partial_ret = new double[suqa::blocks];
    double *dev_partial_ret;
    cudaMalloc((void**)&dev_partial_ret, suqa::blocks*sizeof(double));  
    kernel_suqa_prob1<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, state.data, state.size(), q, rdoub);

    cudaMemcpy(host_partial_ret,dev_partial_ret,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dev_partial_ret); 
    
    for(uint bid=0; bid<suqa::blocks && prob1<rdoub; ++bid){
        prob1 += host_partial_ret[bid]; 
    } 
    delete [] host_partial_ret;
#endif

    c = (uint)(rdoub < prob1); // prob1=1 -> c = 1 surely
    uint c_conj = c^1U; // 1U-c, since c=0U or 1U


//    DEBUG_CALL(std::cout<<"prob1="<<prob1<<", c_conj="<<c_conj<<std::endl);
//#if defined(CUDA_HOST)
//    DEBUG_CALL(std::cout<<"before flipping qbit "<<c_conj<<std::endl);
//    DEBUG_CALL(sparse_print((double*)state.data, state.size()));
//#endif

    // set to 0 coeffs with bm_acc 1-c
    set_ampl_to_zero(state, q, c_conj);
    suqa::vnormalize(state);
}

void suqa::apply_reset(ComplexVec& state, uint q, double rdoub){
//    DEBUG_CALL(std::cout<<"Calling apply_reset() with q="<<q<<"and rdoub="<<rdoub<<std::endl);
    uint c;
    suqa::measure_qbit(state, q, c, rdoub);
    if(c){ // c==1U
        suqa::apply_x(state, q);
        // suqa::vnormalize(state); // normalization shoud be guaranteed by the measure
    }
}  

// faked reset
//void suqa::apply_reset(ComplexVec& state, uint q, double rdoub){
//    for(uint i = 0U; i < state.size(); ++i){
//        if((i >> q) & 1U){ // checks q-th digit in i
//            uint j = i & ~(1U << q); // j has 0 on q-th digit
//            state[j]+=state[i];
//            state[i].x = 0.0;
//            state[i].y = 0.0;
//        }
//    }
//}

void suqa::apply_reset(ComplexVec& state, std::vector<uint> qs, std::vector<double> rdoubs){
    // qs.size() == rdoubs.size()
    for(uint i=0; i<qs.size(); ++i){
        suqa::apply_reset(state, qs[i], rdoubs[i]); 
    } 
}
