#include "io.hpp"

template <typename T>
std::ostream& operator<<(std::ostream& s, const std::complex<T>& c){
    s<<"("<<real(c)<<", "<<imag(c)<<")";
    return s;
}


template <typename T>
void sparse_print(std::vector<std::complex<T>> v){
    for(uint i=0; i<v.size(); ++i){
        if(norm(v[i])>1e-10)
            std::cout<<"i="<<i<<" -> "<<v[i]<<"; ";
    }
    std::cout<<std::endl;
}

void sparse_print(double *v, uint size){
    // for contiguous even-odd entries corresponding to real and imag parts
    for(uint i=0; i<size; ++i){
        std::complex<double> var(v[i*2],v[i*2+1]);
        if(norm(var)>1e-10)
            std::cout<<"i="<<i<<" -> "<<var<<"; ";
    }
    std::cout<<std::endl;
}

void sparse_print(double *v_re, double *v_im, uint size){
    // for non-contiguous even-odd entries corresponding to real and imag parts
    for(uint i=0; i<size; ++i){
        std::complex<double> var(v_re[i],v_im[i]);
       if(norm(var)>1e-10)
            printf("i=%u -> (%.12e, %.12e)\n",i,v_re[i],v_im[i]);
    }
    std::cout<<std::endl;
}

template<class T>
void print(std::vector<T> v){
    for(const auto& el : v)
        std::cout<<el<<" ";
    std::cout<<std::endl;
}

