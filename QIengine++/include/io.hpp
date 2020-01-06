#pragma once
#include <iostream>
#include <complex>
#include <vector>

#ifndef NDEBUG
    #define DEBUG_CALL(x) x
#else
    #define DEBUG_CALL(x)
#endif


template <typename T>
std::ostream& operator<<(std::ostream& s, const std::complex<T>& c){
    s<<"("<<real(c)<<", "<<imag(c)<<")";
    return s;
}


template <typename T>
void sparse_print(std::vector<std::complex<T>> v){
    for(uint i=0; i<v.size(); ++i){
        if(norm(v[i])>1e-8)
            std::cout<<"i="<<i<<" -> "<<v[i]<<"; ";
    }
    std::cout<<std::endl;
}

void sparse_print(double *v, uint size){
    for(uint i=0; i<size; ++i){
        std::complex<double> var(v[i*2],v[i*2+1]);
        if(norm(var)>1e-8)
            std::cout<<"i="<<i<<" -> "<<var<<"; ";
    }
    std::cout<<std::endl;
}

template<class T>
void print(std::vector<T> v){
    for(const auto& el : v)
        std::cout<<el<<" ";
    std::cout<<std::endl;
}
