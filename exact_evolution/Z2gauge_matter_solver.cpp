#include <iostream>
#include <fstream>
#include <cstdio>
#include <complex>
#include <string>
#include <cstring>
#include <vector>
#include <omp.h>
#define ARMA_USE_LAPACK
#define ARMA_USE_OPENMP
#define ARMA_USE_SUPERLU
#include <armadillo>

#define Dim 1<<7
#define K1 dim1
#define K2 (dim1*K1)
#define K3 (dim1*K2)

#define DONE "\033[1;32m[DONE]\033[0m"

using namespace std;
using namespace arma;
typedef complex<double> Complex;
const Complex iu(0, 1);
const int dimstate = 7;


wall_clock timer, tglob;

void dec_to_bin(int dec_state, vector<int>& bin_state, int dimstate){
    for(int i=0;i<dimstate; ++i){
        bin_state[dimstate-i] = dec_state % 2;
        dec_state /= 2;
    }
}


void bin_to_dec(vector<int>& bin_state, int* dec_state, int dimstate){
    *dec_state=0;
    for(int i=0;i<dimstate; ++i){
        *dec_state += bin_state[dimstate-i]*(int)pow(2,i);
//        printf("partial = %d %d\n", (int) pow(2,i), bin_state[dimstate-i]); 
    }
}



int main(){
    
    vector<int> bin_state(dimstate);
    int dec_state = 5;

    dec_to_bin(dec_state, bin_state, dimstate);
    printf("dec = %d\n", dec_state);
    
    
    for(int i=0; i<dimstate; ++i){
        printf("%d", bin_state[i]);     
    }
    printf("\n");
    
    bin_to_dec(bin_state, &dec_state, dimstate);
    printf("decimal again = %d\n", dec_state);




    return 0;
}
