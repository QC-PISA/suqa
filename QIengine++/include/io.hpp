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

