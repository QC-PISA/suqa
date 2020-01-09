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
__global__ void kernel_suqa_vnorm(double *dev_partial_ret_ptr, double *v_comp, uint len){
    extern __shared__ double local_ret[];
    uint tid =  threadIdx.x;
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

    local_ret[tid] = 0.0; // + norm(v[i+blockDim.x]);
    while(i<len){
//        printf("v[%d] = (%.16lg, %.16lg)\n",i, v_re[i], v_im[i]);
        local_ret[tid] += v_comp[i]*v_comp[i];
//        printf("local_ret[%d] = %.10lg\n",tid, local_ret[tid]);
        i += gridDim.x*blockDim.x;
    }
    __syncthreads();

    for(uint s=blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            local_ret[tid] += local_ret[tid+s];
        }
        __syncthreads();
    }
    if(tid<32){
        local_ret[tid] += local_ret[tid+32];
        local_ret[tid] += local_ret[tid+16];
        local_ret[tid] += local_ret[tid+ 8];
        local_ret[tid] += local_ret[tid+ 4];
        local_ret[tid] += local_ret[tid+ 2];
        local_ret[tid] += local_ret[tid+ 1];
    }

    if(tid==0) dev_partial_ret_ptr[blockIdx.x] = local_ret[0];
}

double *host_partial_ret;
double *dev_partial_ret;
double suqa::vnorm(const ComplexVec& v){
    
    double ret = 0.0;
//    double *host_partial_ret = new double[suqa::blocks];
    
    kernel_suqa_vnorm<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double),suqa::stream1>>>(dev_partial_ret, v.data_re, v.size());
    kernel_suqa_vnorm<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double),suqa::stream2>>>(dev_partial_ret+suqa::blocks,  v.data_im, v.size());
    cudaDeviceSynchronize();

    cudaMemcpy(host_partial_ret,dev_partial_ret,2*suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost);
//    cudaDeviceSynchronize();
    
    for(uint bid=0; bid<2*suqa::blocks; ++bid){
//        printf("host_partial_ret[%d(/%d)] = %.10lg\n",bid, suqa::blocks,host_partial_ret[bid]);
        ret += host_partial_ret[bid]; 
    } 
    return sqrt(ret);
}

//__launch_bounds__(128, 6)
__global__ void kernel_suqa_vnormalize_by(double *v_comp, uint len, double value){
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;
    while(i < len){
        v_comp[i]*=value;
        i += gridDim.x*blockDim.x;
    }
}

void suqa::vnormalize(ComplexVec& v){
    double vec_norm = suqa::vnorm(v);
#ifndef NDEBUG
    std::cout<<"vec_norm = "<<vec_norm<<std::endl;
#endif
    // using the inverse, since division is not built-in in cuda
    kernel_suqa_vnormalize_by<<<suqa::blocks,suqa::threads, 0, suqa::stream1>>>(v.data_re, v.size(),1./vec_norm);
    kernel_suqa_vnormalize_by<<<suqa::blocks,suqa::threads, 0, suqa::stream2>>>(v.data_im, v.size(),1./vec_norm);
    cudaDeviceSynchronize();
}



//  X GATE


__global__ 
void kernel_suqa_x(double *const state_re, double *const state_im, uint len, uint q){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if(i & (1U << q)){
            uint j = i & ~(1U << q); // j has 0 on q-th digit
            double tmpval = state_re[i];
            state_re[i]=state_re[j];
            state_re[j]=tmpval;
            tmpval = state_im[i];
            state_im[i]=state_im[j];
            state_im[j]=tmpval;
        }
        i+=gridDim.x*blockDim.x;
    }
}


void suqa::apply_x(ComplexVec& state, uint q){
    kernel_suqa_x<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q);
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
    for(const auto& q : qs)
        kernel_suqa_x<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q);
}  
//void suqa::qi_x(ComplexVec& state, const vector<uint>& qs){
//    for(const auto& q : qs)
//        qi_x(state, q);
//}  

//  HADAMARD GATE

__global__ 
void kernel_suqa_h(double *state_re, double *state_im, uint len, uint q){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    uint i_0 = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i_0<len){
        if((i_0 & (1U << q)) == 0U){
            const uint i_1 = i_0 | (1U << q);
            double a_0_re = state_re[i_0];
            double a_1_re = state_re[i_1];
            double a_0_im = state_im[i_0];
            double a_1_im = state_im[i_1];
            
            state_re[i_0]= TWOSQINV*(a_0_re+a_1_re);
            state_re[i_1]= TWOSQINV*(a_0_re-a_1_re);
            state_im[i_0]= TWOSQINV*(a_0_im+a_1_im);
            state_im[i_1]= TWOSQINV*(a_0_im-a_1_im);
//            state[i_0] = cuCmul(TWOSQINV_CMPX,cuCadd(a_0,a_1));
//            state[i_1] = cuCmul(TWOSQINV_CMPX,cuCsub(a_0,a_1));
        }
        i_0+=gridDim.x*blockDim.x;
    }
}


void suqa::apply_h(ComplexVec& state, uint q){
    kernel_suqa_h<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q);
}  


void suqa::apply_h(ComplexVec& state, const std::vector<uint>& qs){
    for(const auto& q : qs){
        kernel_suqa_h<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q);
    }
}  

//  CONTROLLED-NOT GATE

__global__ 
void kernel_suqa_mcx(double *const state_re, double *const state_im, uint len, uint control_mask, uint mask_qs, uint q_target){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i & control_mask) == mask_qs){
            uint j = i & ~(1U << q_target);
            double tmpval = state_re[i];
            state_re[i]=state_re[j];
            state_re[j]=tmpval;
            tmpval = state_im[i];
            state_im[i]=state_im[j];
            state_im[j]=tmpval;
        }
        i+=gridDim.x*blockDim.x;
    }
}


void suqa::apply_cx(ComplexVec& state, const uint& q_control, const uint& q_target, const uint& q_mask){
    uint mask_qs = 1U << q_target;
    uint mask = mask_qs | (1U << q_control);
    if(q_mask) mask_qs |= (1U << q_control);
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, mask_qs, q_target);
}  

void suqa::apply_mcx(ComplexVec& state, const std::vector<uint>& q_controls, const uint& q_target){
    uint mask = 1U << q_target;
    for(const auto& q : q_controls)
        mask |= 1U << q;
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, mask, q_target);
}  


void suqa::apply_mcx(ComplexVec& state, const std::vector<uint>& q_controls, const std::vector<uint>& q_mask, const uint& q_target){
    uint mask = 1U << q_target;
    for(const auto& q : q_controls)
        mask |= 1U << q;
    uint mask_qs = 1U << q_target;
    for(uint k = 0U; k < q_controls.size(); ++k){
        if(q_mask[k]) mask_qs |= 1U << q_controls[k];
    }

    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, mask_qs, q_target);
}  


__global__ 
void kernel_suqa_swap(double *const state_re, double *const state_im, uint len, uint mask, uint mask_q, uint q2){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i & mask) == mask_q){
            uint j = (i & ~mask_q) | (1U << q2);
            double tmpval = state_re[i];
            state_re[i]=state_re[j];
            state_re[j]=tmpval;
            tmpval = state_im[i];
            state_im[i]=state_im[j];
            state_im[j]=tmpval;
        }
        i+=gridDim.x*blockDim.x;
    }
}

void suqa::apply_swap(ComplexVec& state, const uint& q1, const uint& q2){
    // swap gate: 00->00, 01->10, 10->01, 11->11
    // equivalent to cx(q1,q2)->cx(q2,q1)->cx(q1,q2)
    uint mask_q = (1U << q1);
    uint mask = mask_q | (1U << q2);
    kernel_suqa_swap<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, mask_q, q2);
}

// RESET = measure + classical cx

// sets amplitudes with value <val> in qubit <q> to zero
// !! it leaves the state unnormalized !!
__global__ void kernel_suqa_set_ampl_to_zero(double *state_re, double *state_im, uint len, uint q, uint val){
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;
    while(i<len){
        if(((i >> q) & 1U) == val){
            state_re[i] = 0.0;
            state_im[i] = 0.0;
        }
        i += gridDim.x*blockDim.x;
    }
}


void set_ampl_to_zero(ComplexVec& state, const uint& q, const uint& val){
    kernel_suqa_set_ampl_to_zero<<<suqa::blocks, suqa::threads>>>(state.data_re, state.data_im, state.size(), q, val);
}

//TODO: optimize reduce
__global__ void kernel_suqa_prob1(double *dev_partial_ret, double *v_comp, uint len, uint q, double rdoub){
    extern __shared__ double local_ret[];
    uint tid =  threadIdx.x;
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

    local_ret[tid] = 0.0; // + norm(v[i+blockDim.x]);
    while(i<len){
        if(i & (1U << q)){
            local_ret[tid] += v_comp[i]*v_comp[i];
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
    uint offset = suqa::blocks;
    kernel_suqa_prob1<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double),stream1>>>(dev_partial_ret, state.data_re, state.size(), q, rdoub);
    kernel_suqa_prob1<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double),stream2>>>(dev_partial_ret+offset, state.data_im, state.size(), q, rdoub);
    cudaMemcpyAsync(host_partial_ret,dev_partial_ret,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(host_partial_ret+offset,dev_partial_ret+offset,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost, stream2);
    cudaStreamSynchronize(stream1);
    for(uint bid=0; bid<suqa::blocks && prob1<rdoub; ++bid){
        prob1 += host_partial_ret[bid]; 
    } 
    if(rdoub>prob1){
        cudaStreamSynchronize(stream2);
        for(uint bid=suqa::blocks; bid<2*suqa::blocks && prob1<rdoub; ++bid){
            prob1 += host_partial_ret[bid]; 
        } 
    }
    

    c = (uint)(rdoub < prob1); // prob1=1 -> c = 1 surely
    uint c_conj = c^1U; // 1U-c, since c=0U or 1U


#ifndef NDEBUG
    std::cout<<"prob1="<<prob1<<", c_conj="<<c_conj<<std::endl;
#endif
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

void suqa::setup(){
    cudaHostAlloc((void**)&host_partial_ret,2*suqa::blocks*sizeof(double),cudaHostAllocDefault);
    cudaMalloc((void**)&dev_partial_ret, 2*suqa::blocks*sizeof(double));  
}

void suqa::clear(){
    cudaFree(dev_partial_ret); 
    cudaFreeHost(host_partial_ret);
}
