#include "suqa.cuh"
//#include "cub/cub/cub.cuh" 
//#include <thrust/transform_reduce.h>
//#include <thrust/execution_policy.h>
//#include <thrust/functional.h>

#if !defined(NDEBUG)
double *host_state_re, *host_state_im;
#endif

double *host_partial_ret, *dev_partial_ret;

// global control mask:
// it applies every next operation 
// using it as condition (the user should make sure
// to use it only for operations not involving it)
uint suqa::gc_mask;

void suqa::activate_gc_mask(const bmReg& q_controls){
    suqa::gc_mask=0U;
    for(const auto& q : q_controls)
        suqa::gc_mask |= 1U << q;
}

void suqa::deactivate_gc_mask(){
    suqa::gc_mask=0U;
}

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
__global__ void kernel_suqa_vnorm(double *dev_partial_ret_ptr, double *v_re, double *v_im, uint len){
    extern __shared__ double local_ret[];
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

//    double vj = v_comp[i+(blockDim.x >> 1)];
//    local_ret[tid] =  v_comp[i]*v_comp[i]+vj*vj;
    local_ret[threadIdx.x] = 0.0;
    double tmpval;
    while(i<len){
        tmpval = v_re[i]; 
        local_ret[threadIdx.x] +=  tmpval*tmpval;
        tmpval = v_im[i]; 
        local_ret[threadIdx.x] +=  tmpval*tmpval;
//        if(v_re[i]>0.0)
//            printf("%u %.16lg, %.16lg; loc_ret[%d] = %.16lg\n",i, v_re[i], v_im[i], threadIdx.x, local_ret[threadIdx.x]);
//        tmpval = v_comp[i+blockDim.x]; 
//        local_ret[threadIdx.x] +=  tmpval*tmpval;
        i += gridDim.x*blockDim.x;
//        printf("v[%d] = (%.16lg, %.16lg)\n",i, v_re[i], v_im[i]);
//        printf("local_ret[%d] = %.10lg\n",threadIdx.x, local_ret[threadIdx.x]);

    }
    __syncthreads();

    for(uint s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x < s){
            local_ret[threadIdx.x] += local_ret[threadIdx.x+s];
        }
        __syncthreads();
    }
//    if (blockDim.x >= 1024) { if (threadIdx.x < 512) { local_ret[threadIdx.x] += local_ret[threadIdx.x + 512]; } __syncthreads(); }
//    if (blockDim.x >=  512) { if (threadIdx.x < 256) { local_ret[threadIdx.x] += local_ret[threadIdx.x + 256]; } __syncthreads(); }
//    if (blockDim.x >=  256) { if (threadIdx.x < 128) { local_ret[threadIdx.x] += local_ret[threadIdx.x + 128]; } __syncthreads(); }
//    if (blockDim.x >=  128) { if (threadIdx.x <  64) { local_ret[threadIdx.x] += local_ret[threadIdx.x +  64]; } __syncthreads(); }
//
//    if(threadIdx.x<32){
//        if (blockDim.x >= 64) local_ret[threadIdx.x] += local_ret[threadIdx.x + 32];
//        if (blockDim.x >= 32) local_ret[threadIdx.x] += local_ret[threadIdx.x + 16];
//        if (blockDim.x >= 16) local_ret[threadIdx.x] += local_ret[threadIdx.x +  8];
//        if (blockDim.x >=  8) local_ret[threadIdx.x] += local_ret[threadIdx.x +  4];
//        if (blockDim.x >=  4) local_ret[threadIdx.x] += local_ret[threadIdx.x +  2];
//        if (blockDim.x >=  2) local_ret[threadIdx.x] += local_ret[threadIdx.x +  1];
//    }

    if(threadIdx.x==0){
        dev_partial_ret_ptr[blockIdx.x] = local_ret[0];
//        printf("dev_partial_ret_ptr[%d] = %.16lg\n",blockIdx.x,dev_partial_ret_ptr[blockIdx.x]);
    }
}

double suqa::vnorm(const ComplexVec& v){
    
    double ret = 0.0;
//    double *host_partial_ret = new double[suqa::blocks];
    
    kernel_suqa_vnorm<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, v.data_re, v.data_im, v.size());
//    kernel_suqa_vnorm<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double),suqa::stream1>>>(dev_partial_ret, v.data_re, v.size());
//    kernel_suqa_vnorm<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double),suqa::stream2>>>(dev_partial_ret+suqa::blocks*sizeof(double),  v.data_im, v.size());
//    cudaDeviceSynchronize();

    cudaMemcpy(host_partial_ret,dev_partial_ret,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost);
//    cudaDeviceSynchronize();
    
    for(uint bid=0; bid<suqa::blocks; ++bid){
        
//        printf("host_partial_ret[%d(/%d)] = %.10lg\n",bid, suqa::blocks,host_partial_ret[bid]);
        ret += host_partial_ret[bid]; 
    } 
    return sqrt(ret);
}

////XXX Other possibility: libraries: thrust and cub

////XXX thrust version
//template<typename T>
//struct Square{
// __host__ __device__ __forceinline__
//  T operator()(const T& a) const {
//    return a*a;
//  }
//};
//
//
//template<typename Iterator, typename T, typename UnaryOperation, typename BinaryOperation, typename Pointer>
//__global__ void reduce_kernel(Iterator first, Iterator last, UnaryOperation unary_op, T init, BinaryOperation binary_op, Pointer result){
//  *result = thrust::transform_reduce(thrust::cuda::par, first, last, unary_op, init, binary_op);
//}

//double *ret_re_im, *d_ret_re_im;
//double suqa::vnorm(const ComplexVec& v){
//    
//    double ret = 0.0;
////    double *host_partial_ret = new double[suqa::blocks];
//    Square<double> unary_op;
//    thrust::plus<double> binary_op;
//
////    ret_re_im[0] = thrust::transform_reduce(thrust::cuda::par.on(stream1) , v.data_re, &v.data_re[v.size()-1], unary_op, 0.0, binary_op);
////    ret_re_im[1] = thrust::transform_reduce(thrust::cuda::par.on(stream2), v.data_im, &v.data_im[v.size()-1], unary_op, 0.0, binary_op);
//
//    reduce_kernel<<<1,1,0,stream1>>>(v.data_re, &v.data_re[v.size()-1], unary_op, 0.0, binary_op, &d_ret_re_im[0]);
//    reduce_kernel<<<1,1,0,stream2>>>(v.data_im, v.data_im+v.size(), unary_op, 0.0, binary_op, &d_ret_re_im[1]);
//    cudaMemcpyAsync(ret_re_im+1,d_ret_re_im+1,sizeof(double),cudaMemcpyDeviceToHost, stream2);
//    cudaMemcpyAsync(ret_re_im,d_ret_re_im,sizeof(double),cudaMemcpyDeviceToHost, stream1);
//
//    cudaStreamSynchronize(stream1);
//    cudaStreamSynchronize(stream2);
//    ret +=ret_re_im[0]+ret_re_im[1];
//    
//
////    printf("ret: %.16lg = %.16lg + %.16lg\n",ret, ret_re, ret_im);
//
//
//    return sqrt(ret);
//}

////XXX  CUB version
// double *ret_re_im, *d_ret_re_im;
// double suqa::vnorm(const ComplexVec& v){
//     
//     double ret = 0.0;
// //    double *host_partial_ret = new double[suqa::blocks];
// 
//     cub::TransformInputIterator<double, Square<double>, double*> input_iter_re(v.data_re, Square<double>());
//     cub::TransformInputIterator<double, Square<double>, double*> input_iter_im(v.data_im, Square<double>());
// 
// 
// 
//     void     *d_temp_storage = NULL;
//     size_t   temp_storage_bytes = 0;
//     cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter_re, d_ret_re_im, v.size());
//     // Allocate temporary storage
//     cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
//     // Run sum-reduction
// //    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter_re, &d_ret_re_im[0], v.size(),stream1);
// //    cudaMemcpyAsync(&ret_re_im[0],d_ret_re_im,sizeof(double),cudaMemcpyDeviceToHost,stream1);
// //    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter_im, &d_ret_re_im[1], v.size(),stream2);
// //    cudaMemcpyAsync(&ret_re_im[1],&d_ret_re_im[1],sizeof(double),cudaMemcpyDeviceToHost,stream2); // synchronous
// //    cudaDeviceSynchronize();
//     cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter_re, d_ret_re_im, v.size(),suqa::stream1);
//     cudaStreamSynchronize(stream1);
//     cudaMemcpyAsync(&ret_re_im[0],&d_ret_re_im[0],sizeof(double),cudaMemcpyDeviceToHost,stream1);
//     cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input_iter_im, &d_ret_re_im[1], v.size(),suqa::stream2);
//     cudaStreamSynchronize(stream2);
//     cudaMemcpyAsync(&ret_re_im[1],&d_ret_re_im[1],sizeof(double),cudaMemcpyDeviceToHost,stream2); // synchronous
// 
//     cudaStreamSynchronize(stream1);
//     ret +=ret_re_im[0];
//     cudaStreamSynchronize(stream2);
//     ret +=ret_re_im[1];
// 
//     ret =ret_re_im[0]+ret_re_im[1];
// 
// //    printf("ret: %.16lg = %.16lg + %.16lg\n",ret, ret_re_im[0], ret_re_im[1]);
// 
//     cudaFree(d_temp_storage);
// 
//     return sqrt(ret);
// }

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
    kernel_suqa_vnormalize_by<<<suqa::blocks,suqa::threads>>>(v.data, 2*v.size(),1./vec_norm);
//    kernel_suqa_vnormalize_by<<<suqa::blocks,suqa::threads, 0, suqa::stream1>>>(v.data_re, v.size(),1./vec_norm);
//    kernel_suqa_vnormalize_by<<<suqa::blocks,suqa::threads, 0, suqa::stream2>>>(v.data_im, v.size(),1./vec_norm);
    cudaDeviceSynchronize();
}



//  X GATE


__global__ 
void kernel_suqa_x(double *const state_re, double *const state_im, uint len, uint q, uint glob_mask){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    glob_mask |= (1U <<q);
    while(i<len){
        if((i & glob_mask) == glob_mask){
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
    kernel_suqa_x<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, gc_mask);
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


void suqa::apply_x(ComplexVec& state, const bmReg& qs){
    for(const auto& q : qs)
        kernel_suqa_x<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, gc_mask);
}  
//void suqa::qi_x(ComplexVec& state, const vector<uint>& qs){
//    for(const auto& q : qs)
//        qi_x(state, q);
//}  

//  HADAMARD GATE

__global__ 
void kernel_suqa_h(double *state_re, double *state_im, uint len, uint q, uint glob_mask){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    uint i_0 = blockDim.x*blockIdx.x + threadIdx.x;    
    
    uint loc_mask = glob_mask | (1U << q);
    while(i_0<len){
        if((i_0 & loc_mask) == glob_mask){
            const uint i_1 = i_0 | (1U << q);
            double a_0_re = state_re[i_0];
            double a_1_re = state_re[i_1];
            double a_0_im = state_im[i_0];
            double a_1_im = state_im[i_1];
            
            state_re[i_0]= TWOSQINV*(a_0_re+a_1_re);
            state_re[i_1]= TWOSQINV*(a_0_re-a_1_re);
            state_im[i_0]= TWOSQINV*(a_0_im+a_1_im);
            state_im[i_1]= TWOSQINV*(a_0_im-a_1_im);
        }
        i_0+=gridDim.x*blockDim.x;
    }
}


void suqa::apply_h(ComplexVec& state, uint q){
    kernel_suqa_h<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, gc_mask);
    DEBUG_READ_STATE(state);
}  


void suqa::apply_h(ComplexVec& state, const bmReg& qs){
    for(const auto& q : qs){
        kernel_suqa_h<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, gc_mask);
    }
}  

//  PI/8 GATES

__global__ 
void kernel_suqa_t(double *state_re, double *state_im, uint len, uint q, uint glob_mask){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    uint i_1 = blockDim.x*blockIdx.x + threadIdx.x;    
    glob_mask |= (1U << q);
    while(i_1<len){
        if((i_1 & glob_mask) == glob_mask){
            double a_1_re = state_re[i_1];
            double a_1_im = state_im[i_1];
            
            state_re[i_1]= TWOSQINV*(a_1_re - a_1_im);
            state_im[i_1]= TWOSQINV*(a_1_im + a_1_re);
        }
        i_1+=gridDim.x*blockDim.x;
    }
}

__global__ 
void kernel_suqa_tdg(double *state_re, double *state_im, uint len, uint q, uint glob_mask){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    uint i_1 = blockDim.x*blockIdx.x + threadIdx.x;    
    glob_mask |= (1U << q);
    while(i_1<len){
        if((i_1 & glob_mask) == glob_mask){
            double a_1_re = state_re[i_1];
            double a_1_im = state_im[i_1];
            
            state_re[i_1]= TWOSQINV*(a_1_re + a_1_im);
            state_im[i_1]= TWOSQINV*(a_1_im - a_1_re);
        }
        i_1+=gridDim.x*blockDim.x;
    }
}


// T gate (single qubit)
void suqa::apply_t(ComplexVec& state, uint q){
    kernel_suqa_t<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, gc_mask);
}  
// T gate (multiple qubits)
void suqa::apply_t(ComplexVec& state, const bmReg& qs){
    for(const auto& q : qs){
        kernel_suqa_t<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, gc_mask);
    }
}  

// T^{\dagger} gate (single qubit)
void suqa::apply_tdg(ComplexVec& state, uint q){
    kernel_suqa_tdg<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, gc_mask);
}  
// T^{\dagger} gate (multiple qubits)
void suqa::apply_tdg(ComplexVec& state, const bmReg& qs){
    for(const auto& q : qs){
        kernel_suqa_tdg<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, gc_mask);
    }
}  

// U1 GATE

__global__ 
void kernel_suqa_u1(double *state_re, double *state_im, uint len, uint q, Complex phase, uint glob_mask){
//    const Complex TWOSQINV_CMPX = make_cuDoubleComplex(TWOSQINV,0.0f);
     
    uint i_1 = blockDim.x*blockIdx.x + threadIdx.x;    
    glob_mask |= (1U << q);
    while(i_1<len){
        if((i_1 & glob_mask) == glob_mask){
            double tmpval = state_re[i_1]; 
            state_re[i_1] = state_re[i_1]*phase.x-state_im[i_1]*phase.y;
            state_im[i_1] = tmpval*phase.y+state_im[i_1]*phase.x;

        }
        i_1+=gridDim.x*blockDim.x;
    }
}


void suqa::apply_u1(ComplexVec& state, uint q, double phase){
    Complex phasec;
    sincos(phase, &phasec.y, &phasec.x);
    kernel_suqa_u1<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), q, phasec, gc_mask);
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
    uint mask_qs = (1U << q_target) | gc_mask;
    uint mask = mask_qs | (1U << q_control);
    if(q_mask) mask_qs |= (1U << q_control);
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, mask_qs, q_target);
}  

void suqa::apply_mcx(ComplexVec& state, const bmReg& q_controls, const uint& q_target){
    uint mask = (1U << q_target) | gc_mask;
    for(const auto& q : q_controls)
        mask |= 1U << q;
    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, mask, q_target);
}  


void suqa::apply_mcx(ComplexVec& state, const bmReg& q_controls, const bmReg& q_mask, const uint& q_target){
    uint mask = (1U << q_target) | gc_mask;
    for(const auto& q : q_controls)
        mask |= 1U << q;
    uint mask_qs = (1U << q_target) | gc_mask;
    for(uint k = 0U; k < q_controls.size(); ++k){
        if(q_mask[k]) mask_qs |= 1U << q_controls[k];
    }

    kernel_suqa_mcx<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, mask_qs, q_target);
}  

__global__ 
void kernel_suqa_mcu1(double *const state_re, double *const state_im, uint len, uint control_mask, uint mask_qs, uint q_target, Complex rphase){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i & control_mask) == mask_qs){
//            uint j = i & ~(1U << q_target);
            double tmpval = state_re[i]; 
            state_re[i] = state_re[i]*rphase.x-state_im[i]*rphase.y;
            state_im[i] = tmpval*rphase.y+state_im[i]*rphase.x;
        }
        i+=gridDim.x*blockDim.x;
    }
}

void suqa::apply_cu1(ComplexVec& state, uint q_control, uint q_target, double phase, uint q_mask){
    uint mask_qs = (1U << q_target) | gc_mask;
    uint mask = mask_qs | (1U << q_control);
    if(q_mask) mask_qs |= (1U << q_control);

    Complex phasec;
    sincos(phase, &phasec.y, &phasec.x);

    kernel_suqa_mcu1<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, mask_qs, q_target, phasec);
}

void suqa::apply_mcu1(ComplexVec& state, const bmReg& q_controls, const bmReg& q_mask, const uint& q_target, double phase){
    uint mask = (1U << q_target) | gc_mask;
    for(const auto& q : q_controls)
        mask |= 1U << q;
    uint mask_qs = (1U << q_target) | gc_mask;
    for(uint k = 0U; k < q_controls.size(); ++k){
        if(q_mask[k]) mask_qs |= 1U << q_controls[k];
    }

    Complex phasec;
    sincos(phase, &phasec.y, &phasec.x);

    kernel_suqa_mcu1<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask, mask_qs, q_target, phasec);
}

__global__ 
void kernel_suqa_swap(double *const state_re, double *const state_im, uint len, uint mask00, uint mask11, uint mask_q1, uint mask_q2){
    int i = blockDim.x*blockIdx.x + threadIdx.x;    
    while(i<len){
        if((i & mask11) == mask00){
            // i -> ...00..., i_1 -> ...10..., i_2 -> ...01...
            uint i_1 = i | mask_q1;
            uint i_2 = i | mask_q2;
            double tmpval = state_re[i_1];
            state_re[i_1]=state_re[i_2];
            state_re[i_2]=tmpval;
            tmpval = state_im[i_1];
            state_im[i_1]=state_im[i_2];
            state_im[i_2]=tmpval;
        }
        i+=gridDim.x*blockDim.x;
    }
}

void suqa::apply_swap(ComplexVec& state, const uint& q1, const uint& q2){
    // swap gate: 00->00, 01->10, 10->01, 11->11
    // equivalent to cx(q1,q2)->cx(q2,q1)->cx(q1,q2)
    uint mask00 = gc_mask;
    uint mask11 = mask00;
    uint mask_q1 = (1U << q1);
    uint mask_q2 = (1U << q2);
    mask11 |= mask_q1 | mask_q2;
    kernel_suqa_swap<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask00, mask11, mask_q1, mask_q2);
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

__global__
void kernel_suqa_pauli_TP_rotation_x(double *const state_re, double *const state_im, uint len, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    double tmp_re0, tmp_im0, tmp_re1, tmp_im1;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;


            tmp_re0 = state_re[i_0];
            tmp_im0 = state_im[i_0];
            tmp_re1 = state_re[i_1];
            tmp_im1 = state_im[i_1];
            
            state_re[i_0] = tmp_re0*ctheta - tmp_im1*stheta;
            state_im[i_0] = tmp_im0*ctheta + tmp_re1*stheta;

            state_re[i_1] = tmp_re1*ctheta - tmp_im0*stheta; 
            state_im[i_1] = tmp_im1*ctheta + tmp_re0*stheta;
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_y(double *const state_re, double *const state_im, uint len, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    double tmp_re0, tmp_im0, tmp_re1, tmp_im1;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;


            tmp_re0 = state_re[i_0];
            tmp_im0 = state_im[i_0];
            tmp_re1 = state_re[i_1];
            tmp_im1 = state_im[i_1];
            
            state_re[i_0] = tmp_re0*ctheta + tmp_re1*stheta;
            state_im[i_0] = tmp_im0*ctheta + tmp_im1*stheta;

            state_re[i_1] = tmp_re1*ctheta - tmp_re0*stheta; 
            state_im[i_1] = tmp_im1*ctheta - tmp_im0*stheta;
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_z(double *const state_re, double *const state_im, uint len, uint mask0s, uint mask1s, uint mask_q1, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    double tmp_re0, tmp_im0, tmp_re1, tmp_im1;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;


            tmp_re0 = state_re[i_0];
            tmp_im0 = state_im[i_0];
            tmp_re1 = state_re[i_1];
            tmp_im1 = state_im[i_1];
            
            state_re[i_0] = tmp_re0*ctheta - tmp_im0*stheta;
            state_im[i_0] = tmp_im0*ctheta + tmp_re0*stheta;

            state_re[i_1] = tmp_re1*ctheta + tmp_im1*stheta; 
            state_im[i_1] = tmp_im1*ctheta - tmp_re1*stheta;
        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_zxx(double *const state_re, double *const state_im, uint len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, uint mask_q3, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    double tmpval;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            uint i_4 = i_0 | mask_q3;
            uint i_5 = i_4 | i_1;
            uint i_6 = i_4 | i_2;
            uint i_7 = i_4 | i_3;
            
            // 0<->3
            tmpval = state_re[i_0];
            state_re[i_0] = tmpval*ctheta -state_im[i_3]*stheta;
            state_im[i_3] = state_im[i_3]*ctheta +tmpval*stheta;

            tmpval = state_im[i_0]; state_im[i_0] = tmpval*ctheta +state_re[i_3]*stheta;
            state_re[i_3] = state_re[i_3]*ctheta -tmpval*stheta;

            // 5<->6
            tmpval = state_re[i_5];
            state_re[i_5] = tmpval*ctheta +state_im[i_6]*stheta;
            state_im[i_6] = state_im[i_6]*ctheta -tmpval*stheta;

            tmpval = state_im[i_5];
            state_im[i_5] = tmpval*ctheta -state_re[i_6]*stheta;
            state_re[i_6] = state_re[i_6]*ctheta +tmpval*stheta;

            // 1<->2
            tmpval = state_re[i_1];
            state_re[i_1] = tmpval*ctheta -state_im[i_2]*stheta;
            state_im[i_2] = state_im[i_2]*ctheta +tmpval*stheta;

            tmpval = state_im[i_1];
            state_im[i_1] = tmpval*ctheta +state_re[i_2]*stheta;
            state_re[i_2] = state_re[i_2]*ctheta -tmpval*stheta;


            // 4<->7
            tmpval = state_re[i_4];
            state_re[i_4] = tmpval*ctheta +state_im[i_7]*stheta;
            state_im[i_7] = state_im[i_7]*ctheta -tmpval*stheta;

            tmpval = state_im[i_4];
            state_im[i_4] = tmpval*ctheta -state_re[i_7]*stheta;
            state_re[i_7] = state_re[i_7]*ctheta +tmpval*stheta;

        }
        i_0+=gridDim.x*blockDim.x;
    }
}

__global__
void kernel_suqa_pauli_TP_rotation_zyy(double *const state_re, double *const state_im, uint len, uint mask0s, uint mask1s, uint mask_q1, uint mask_q2, uint mask_q3, double ctheta, double stheta){
    int i_0 = blockDim.x*blockIdx.x + threadIdx.x;
    double tmpval;
    while(i_0<len){
        if((i_0 & mask1s) == mask0s){
            // i -> ...00..., i_1 -> ...01..., i_2 -> ...10...
            uint i_1 = i_0 | mask_q1;
            uint i_2 = i_0 | mask_q2;
            uint i_3 = i_2 | i_1;
            uint i_4 = i_0 | mask_q3;
            uint i_5 = i_4 | i_1;
            uint i_6 = i_4 | i_2;
            uint i_7 = i_4 | i_3;

            
            // 0<->3
            tmpval = state_re[i_0];
            state_re[i_0] = tmpval*ctheta +state_im[i_3]*stheta;
            state_im[i_3] = state_im[i_3]*ctheta -tmpval*stheta;

            tmpval = state_im[i_0];
            state_im[i_0] = tmpval*ctheta -state_re[i_3]*stheta;
            state_re[i_3] = state_re[i_3]*ctheta +tmpval*stheta;

            // 5<->6
            tmpval = state_re[i_5];
            state_re[i_5] = tmpval*ctheta +state_im[i_6]*stheta;
            state_im[i_6] = state_im[i_6]*ctheta -tmpval*stheta;

            tmpval = state_im[i_5];
            state_im[i_5] = tmpval*ctheta -state_re[i_6]*stheta;
            state_re[i_6] = state_re[i_6]*ctheta +tmpval*stheta;

            // 1<->2
            tmpval = state_re[i_1];
            state_re[i_1] = tmpval*ctheta -state_im[i_2]*stheta;
            state_im[i_2] = state_im[i_2]*ctheta +tmpval*stheta;

            tmpval = state_im[i_1];
            state_im[i_1] = tmpval*ctheta +state_re[i_2]*stheta;
            state_re[i_2] = state_re[i_2]*ctheta -tmpval*stheta;

            // 4<->7
            tmpval = state_re[i_4];
            state_re[i_4] = tmpval*ctheta -state_im[i_7]*stheta;
            state_im[i_7] = state_im[i_7]*ctheta +tmpval*stheta;

            tmpval = state_im[i_4];
            state_im[i_4] = tmpval*ctheta +state_re[i_7]*stheta;
            state_re[i_7] = state_re[i_7]*ctheta -tmpval*stheta;

        }
        i_0+=gridDim.x*blockDim.x;
    }
}


// rotation by phase in the direction of a pauli tensor product
void suqa::apply_pauli_TP_rotation(ComplexVec& state, const bmReg& q_apply, const std::vector<uint>& pauli_TPtype, double phase){
    uint mask0s = gc_mask;
    uint mask1s = mask0s;
    uint mask_q1, mask_q2, mask_q3;
    for(const auto& q : q_apply){
        mask1s |= (1U << q);
    }
    double sph, cph;
    sincos(phase, &sph, &cph);

    if(q_apply.size()!=pauli_TPtype.size()){
        throw std::runtime_error("ERROR: in suqa::apply_pauli_TP_rotation(): mismatch between qubits number and pauli types specified");
    }

    if(q_apply.size()==1U){
        mask_q1 = (1U << q_apply[0]);
        switch(pauli_TPtype[0]){
            case PAULI_X:
                kernel_suqa_pauli_TP_rotation_x<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask0s, mask1s, mask_q1, cph, sph);
                break;
            case PAULI_Y:
                kernel_suqa_pauli_TP_rotation_y<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask0s, mask1s, mask_q1, cph, sph);
                break;
            case PAULI_Z:
                kernel_suqa_pauli_TP_rotation_z<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask0s, mask1s, mask_q1, cph, sph);
                break;
            default:
                break;
        }
    }else if(q_apply.size()==3U){
        int i_z = -1, i1, i2;
        if(pauli_TPtype[0]==PAULI_Z){ 
            i_z=0;
            i1=1;
            i2=2;
        }else if(pauli_TPtype[1]==PAULI_Z){
            i_z=1;
            i1=0;
            i2=2;
        }else if(pauli_TPtype[2]==PAULI_Z){
            i_z=2;
            i1=0;
            i2=1;
        }else{
            throw std::runtime_error("ERROR: unimplemented pauli TP rotation with 3 qubits in the selected configuration");
        }
        mask_q3 = (1U << q_apply[i_z]);
        mask_q1 = (1U << q_apply[i1]);
        mask_q2 = (1U << q_apply[i2]);
        if(pauli_TPtype[i1]==PAULI_X and pauli_TPtype[i2]==PAULI_X){
                kernel_suqa_pauli_TP_rotation_zxx<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask0s, mask1s, mask_q1, mask_q2, mask_q3, cph, sph);
        }else if(pauli_TPtype[i1]==PAULI_Y and pauli_TPtype[i2]==PAULI_Y){
                kernel_suqa_pauli_TP_rotation_zyy<<<suqa::blocks,suqa::threads>>>(state.data_re, state.data_im, state.size(), mask0s, mask1s, mask_q1, mask_q2, mask_q3, cph, sph);
        }else{
            throw std::runtime_error("ERROR: unimplemented pauli TP rotation with 3 qubits in the selected configuration");
        }
    }else{
        throw std::runtime_error(("ERROR: unimplemented pauli tensor product rotation with "+std::to_string(q_apply.size())+" qubits").c_str());
    }
}




void set_ampl_to_zero(ComplexVec& state, const uint& q, const uint& val){
    kernel_suqa_set_ampl_to_zero<<<suqa::blocks, suqa::threads>>>(state.data_re, state.data_im, state.size(), q, val);
}

__global__ void kernel_suqa_prob1(double *dev_partial_ret_ptr, double *v_re, double *v_im, uint len, uint q){
    extern __shared__ double local_ret[];
    uint tid = threadIdx.x;
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

//    double vj = v_comp[i+(blockDim.x >> 1)];
//    local_ret[tid] =  v_comp[i]*v_comp[i]+vj*vj;
    local_ret[threadIdx.x] = 0.0;
    double tmpval;
    while(i<len){
        if(i & (1U << q)){
            tmpval = v_re[i];
            local_ret[tid] +=  tmpval*tmpval;
            tmpval = v_im[i];
            local_ret[tid] +=  tmpval*tmpval;
        }
        i += gridDim.x*blockDim.x;
//        printf("v[%d] = (%.16lg, %.16lg)\n",i, v_re[i], v_im[i]);
//        printf("local_ret[%d] = %.10lg\n",tid, local_ret[tid]);

    }
    __syncthreads();

    for(uint s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            local_ret[tid] += local_ret[tid+s];
        }
        __syncthreads();
    }
//    if (blockDim.x >= 1024) { if (tid < 512) { local_ret[tid] += local_ret[tid + 512]; } __syncthreads(); }
//    if (blockDim.x >=  512) { if (tid < 256) { local_ret[tid] += local_ret[tid + 256]; } __syncthreads(); }
//    if (blockDim.x >=  256) { if (tid < 128) { local_ret[tid] += local_ret[tid + 128]; } __syncthreads(); }
//    if (blockDim.x >=  128) { if (tid <  64) { local_ret[tid] += local_ret[tid +  64]; } __syncthreads(); }
//
//    if(tid<32){
//        if (blockDim.x >= 64) local_ret[tid] += local_ret[tid + 32];
//        if (blockDim.x >= 32) local_ret[tid] += local_ret[tid + 16];
//        if (blockDim.x >= 16) local_ret[tid] += local_ret[tid +  8];
//        if (blockDim.x >=  8) local_ret[tid] += local_ret[tid +  4];
//        if (blockDim.x >=  4) local_ret[tid] += local_ret[tid +  2];
//        if (blockDim.x >=  2) local_ret[tid] += local_ret[tid +  1];
//    }

    if(tid==0) dev_partial_ret_ptr[blockIdx.x] = local_ret[0];
}


void suqa::measure_qbit(ComplexVec& state, uint q, uint& c, double rdoub){
    double prob1 = 0.0;
    c=0U;
//    kernel_suqa_prob1<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, state.data, 2*state.size(), q);
    kernel_suqa_prob1<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, state.data_re, state.data_im, state.size(), q);
//    kernel_suqa_prob1<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double),stream2>>>(dev_partial_ret+blocks, state.data_im, state.size(), q);
//    cudaDeviceSynchronize();
    cudaMemcpy(host_partial_ret,dev_partial_ret,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost);
    
    for(uint bid=0; bid<suqa::blocks && prob1<rdoub; ++bid){
        prob1 += host_partial_ret[bid]; 
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

////TODO: can be optimized for multiple qbits measures?
void suqa::measure_qbits(ComplexVec& state, const bmReg& qs, std::vector<uint>& cs,const std::vector<double>& rdoubs){
    for(uint k = 0U; k < qs.size(); ++k)
        suqa::measure_qbit(state, qs[k], cs[k], rdoubs[k]);
}


__global__ void kernel_suqa_prob_filter(double *dev_partial_ret_ptr, double *v_re, double *v_im, uint len, uint mask_qs, uint mask){
    extern __shared__ double local_ret[];
    uint tid = threadIdx.x;
    uint i =  blockIdx.x*blockDim.x + threadIdx.x;

//    double vj = v_comp[i+(blockDim.x >> 1)];
//    local_ret[tid] =  v_comp[i]*v_comp[i]+vj*vj;
    local_ret[threadIdx.x] = 0.0;
    double tmpval;
    while(i<len){
        if((i & mask_qs) == mask){
            tmpval = v_re[i];
            local_ret[tid] +=  tmpval*tmpval;
            tmpval = v_im[i];
            local_ret[tid] +=  tmpval*tmpval;
        }
        i += gridDim.x*blockDim.x;
//        printf("v[%d] = (%.16lg, %.16lg)\n",i, v_re[i], v_im[i]);
//        printf("local_ret[%d] = %.10lg\n",tid, local_ret[tid]);

    }
    __syncthreads();

    for(uint s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            local_ret[tid] += local_ret[tid+s];
        }
        __syncthreads();
    }
//    if (blockDim.x >= 1024) { if (tid < 512) { local_ret[tid] += local_ret[tid + 512]; } __syncthreads(); }
//    if (blockDim.x >=  512) { if (tid < 256) { local_ret[tid] += local_ret[tid + 256]; } __syncthreads(); }
//    if (blockDim.x >=  256) { if (tid < 128) { local_ret[tid] += local_ret[tid + 128]; } __syncthreads(); }
//    if (blockDim.x >=  128) { if (tid <  64) { local_ret[tid] += local_ret[tid +  64]; } __syncthreads(); }
//
//    if(tid<32){
//        if (blockDim.x >= 64) local_ret[tid] += local_ret[tid + 32];
//        if (blockDim.x >= 32) local_ret[tid] += local_ret[tid + 16];
//        if (blockDim.x >= 16) local_ret[tid] += local_ret[tid +  8];
//        if (blockDim.x >=  8) local_ret[tid] += local_ret[tid +  4];
//        if (blockDim.x >=  4) local_ret[tid] += local_ret[tid +  2];
//        if (blockDim.x >=  2) local_ret[tid] += local_ret[tid +  1];
//    }

    if(tid==0) dev_partial_ret_ptr[blockIdx.x] = local_ret[0];
}



void suqa::prob_filter(ComplexVec& state, const bmReg& qs, const std::vector<uint>& q_mask, double &prob){
    prob = 0.0;
    uint mask_qs = 0U;
    for(const auto& q : qs)
        mask_qs |= 1U << q;
    uint mask = 0U;
    for(uint k = 0U; k < q_mask.size(); ++k){
        if(q_mask[k]) mask |= q_mask[k] << qs[k];
    }
//    kernel_suqa_prob_filter<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, state.data, 2*state.size(), q);
    kernel_suqa_prob_filter<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret, state.data_re, state.data_im, state.size(), mask_qs, mask);
//    kernel_suqa_prob_filter<<<suqa::blocks,suqa::threads,suqa::threads*sizeof(double)>>>(dev_partial_ret+suqa::blocks, state.data_im, state.size(), mask_qs, mask);
//    cudaDeviceSynchronize();
    cudaMemcpy(host_partial_ret,dev_partial_ret,suqa::blocks*sizeof(double), cudaMemcpyDeviceToHost);
    
    for(uint bid=0; bid<suqa::blocks; ++bid){
        prob += host_partial_ret[bid]; 
    } 
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

// fake reset
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

void suqa::apply_reset(ComplexVec& state, const bmReg& qs, std::vector<double> rdoubs){
    // qs.size() == rdoubs.size()
    for(uint i=0; i<qs.size(); ++i){
        suqa::apply_reset(state, qs[i], rdoubs[i]); 
    } 
}

void suqa::setup(uint Dim){
    cudaHostAlloc((void**)&host_partial_ret,suqa::blocks*sizeof(double),cudaHostAllocDefault);
    cudaMalloc((void**)&dev_partial_ret, suqa::blocks*sizeof(double));  
// the following are allocated only for library versions of reduce
//    cudaHostAlloc((void**)&ret_re_im,2*sizeof(double),cudaHostAllocDefault);
//    cudaMalloc((void**)&d_ret_re_im,2*sizeof(double));

    cudaDeviceProp prop;
    int whichDevice;
    HANDLE_CUDACALL( cudaGetDevice( &whichDevice ) );
    HANDLE_CUDACALL( cudaGetDeviceProperties( &prop, whichDevice ) );

    HANDLE_CUDACALL( cudaStreamCreate( &suqa::stream1 ) );
    if (!prop.deviceOverlap) {
        DEBUG_CALL(printf( "Device will not handle overlaps, so no "
        "speed up from streams\n" ));
        suqa::stream2 = suqa::stream1;
    }else{
        HANDLE_CUDACALL( cudaStreamCreate( &suqa::stream2 ) );
    }

#if !defined(NDEBUG)
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_re,Dim*sizeof(double),cudaHostAllocDefault));
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_im,Dim*sizeof(double),cudaHostAllocDefault));
#endif
}

void suqa::clear(){
//    cudaFree(d_ret_re_im);
//    cudaFreeHost(ret_re_im);
    cudaFree(dev_partial_ret); 
    cudaFreeHost(host_partial_ret);

#ifndef NDEBUG
    HANDLE_CUDACALL(cudaFreeHost(host_state_re));
    HANDLE_CUDACALL(cudaFreeHost(host_state_im));
#endif

    HANDLE_CUDACALL( cudaStreamDestroy( suqa::stream1 ) );
    if (suqa::stream1!=suqa::stream2) {
        HANDLE_CUDACALL( cudaStreamDestroy( suqa::stream2 ) );
    }

}
