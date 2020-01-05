#include "suqa.cuh"

//TODO: optimize reduce by unrolling
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

double suqa::vnorm(uint blocks, uint threads, const ComplexVec& v){
    double ret = 0.0;
#if defined(CUDA_HOST)
    for(uint i=0; i<v.size(); ++i){
        ret += norm(v[i]);
    }
    
    return sqrt(ret);
#else // CUDA defined
    double *dev_partial_ret;
    double *host_partial_ret = new double[blocks];
    cudaMalloc((void**)&dev_partial_ret, blocks*sizeof(double));  
    kernel_suqa_vnorm<<<blocks,threads,threads*sizeof(double)>>>
        (dev_partial_ret, v.data, v.size());

    cudaMemcpy(host_partial_ret,dev_partial_ret,blocks*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dev_partial_ret); 
    
    for(uint bid=0; bid<blocks; ++bid){
        ret += host_partial_ret[bid]; 
    } 
    delete [] host_partial_ret;
#endif
    return sqrt(ret);
}

__global__ void kernel_suqa_vnormalize_by(Complex *v, uint len, double value){
    uint tid =  blockDim.x*gridDim.x + threadIdx.x;
    while( tid < len){
        v[tid]/=value;
        tid += blockDim.x*gridDim.x;
    }
}

void suqa::vnormalize(uint blocks, uint threads, ComplexVec& v){
    double vec_norm = suqa::vnorm(blocks, threads, v);
#if defined(CUDA_HOST)
    for(uint i=0; i<v.size(); ++i){
        v[i]/=vec_norm;
    }
#else // CUDA defined
    kernel_suqa_vnormalize_by<<<blocks,threads>>>
        (v.data, v.size(),vec_norm);
#endif
}


//void suqa::qi_reset(ComplexVec& state, const uint& q){
//#if !defined(CUDA) || !defined(CUDA_DEVICE)
//    for(uint i = 0U; i < state.size(); ++i){
//        if((i >> q) & 1U){ // checks q-th digit in i
//            uint j = i & ~(1U << q); // j has 0 on q-th digit
//            state[j]+=state[i];
//            state[i]= {0.0, 0.0};
//        }
//    }
//    suqa::vnormalize(state);
//
//#else
//    //TODO: implement code for device
//#endif
//}  
//
//void suqa::qi_reset(ComplexVec& state, const vector<uint>& qs){
//#if !defined(CUDA) || !defined(CUDA_DEVICE)
//    uint mask=0U;
//    for(const auto& q : qs)
//        mask |= 1U << q;
//
//    for(uint i = 0U; i < state.size(); ++i){
//        if((i & mask) != 0U){ // checks q-th digit in i
//            state[i & ~mask]+=state[i];
//            state[i]= 0.0;
//        }
//    }
//    suqa::vnormalize(state);
//
//#else
//    //TODO: implement code for device
//#endif
//}  

//  X GATE

__host__ __device__
void swap_cmpx(Complex *const a, Complex *const b){
    Complex tmp_c = *a;
    *a = *b;
    *b = tmp_c;
}


__global__ 
void kernel_suqa_x(Complex *const state, uint len, uint q){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i >> q) & 1U){
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            swap_cmpx(&state[i],&state[j]);
        }
        i+=gridDim.x*blockDim.x;
    }
}


void suqa::apply_x(uint blocks, uint threads, ComplexVec& state, uint q){
#if defined(CUDA_HOST)
    for(uint i=0; i<state.size(); ++i){
        if((i >> q) & 1U){
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            swap_cmpx(&state[i],&state[j]);
        }
    }
#else // CUDA defined
    kernel_suqa_x<<<blocks,threads>>>
        (state.data, state.size(), q);
#endif
}  


//  HADAMARD GATE

__global__ 
void kernel_suqa_h(Complex *const state, uint len, uint q){
    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i_0<len){
        if((i_0 & (1U << q)) == 0U){
            const int i_1 = i_0 | (1U << q);
            Complex a_0 = state[i_0];
            Complex a_1 = state[i_1];
            
            state[i_0] = cuCmul(TWOSQINV_CMPX,cuCadd(a_0,a_1));
            state[i_1] = cuCmul(TWOSQINV_CMPX,cuCsub(a_0,a_1));
        }
        i_0+=gridDim.x*blockDim.x;
    }
}


void suqa::apply_h(uint blocks, uint threads, ComplexVec& state, uint q){
#if defined(CUDA_HOST)
    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
    for(uint i_0=0; i_0<state.size(); ++i_0){
        if((i_0 & (1U << q)) == 0U){
            const int i_1 = i_0 | (1U << q);
            Complex a_0 = state[i_0];
            Complex a_1 = state[i_1];
            
            state[i_0] = cuCmul(TWOSQINV_CMPX,cuCadd(a_0,a_1));
            state[i_1] = cuCmul(TWOSQINV_CMPX,cuCsub(a_0,a_1));
        }
    }
#else // CUDA defined
    kernel_suqa_h<<<blocks,threads>>>
        (state.data, state.size(), q);
#endif
}  


//  CONTROLLED-NOT GATE

__global__ 
void kernel_suqa_cx(Complex *const state, uint len, uint q_control, uint q_target){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
            uint j = i & ~(1U << q_target);
            swap_cmpx(&state[i],&state[j]);
        }
        i+=gridDim.x*blockDim.x;
    }
}

void suqa::apply_cx(uint blocks, uint threads, ComplexVec& state, uint q_control, uint q_target){
#if defined(CUDA_HOST)
    for(uint i=0; i<state.size(); ++i){
        if(((i >> q_control) & 1U) && ((i >> q_target) & 1U)){
            uint j = i & ~(1U << q_target);
            swap_cmpx(&state[i],&state[j]);
        }
    }
#else // CUDA defined
    kernel_suqa_cx<<<blocks,threads>>>
        (state.data, state.size(), q_control, q_target);
#endif
}  

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
