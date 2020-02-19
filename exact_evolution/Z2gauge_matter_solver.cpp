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

#define DONE "\033[1;32m[DONE]\033[0m"

using namespace std;
using namespace arma;
typedef complex<double> Complex;
const Complex iu(0, 1);

const int dimstate = 7; // number of qubits.
const int num_ferm = 4; // number of (staggered) fermions.
const int num_link = 3; // number of z2 gauge links.
const int Dim = (int)pow(2,dimstate); //length of the state.

wall_clock timer, tglob;

// Convert from decimal to binary. The form is |F4F3F2F1>|G3G2G1>. F is the fermion variable, G the link.
void dec_to_bin(int dec_state, vector<int>& bin_state, int dimstate){
    for(int i=0;i<dimstate; ++i){
        bin_state[dimstate-i-1] = dec_state % 2;
        dec_state /= 2;
    }
}

// Convert from binary to decimal. The most significant Qubit is the one on the right.
int bin_to_dec(vector<int>& bin_state, int dimstate){
    int dec_state=0;
    for(int i=0;i<dimstate; ++i){
        dec_state += bin_state[dimstate-i-1]*(int)pow(2,i);
    }
    return dec_state;
}

//compute the element <i|H_m|j>. Where H_m = \sum_{i=1}^{num_ferm} -\frac{m}{2}(-1)^i\sigma^{(z)}_i --- i is the site of the lattice.
double H_mass_ferm(vector<int>& bin_state_i, vector<int>& bin_state_j, double mass){
    vector<int> ferm_part_i(num_ferm);
    vector<int> ferm_part_j(num_ferm);

    for(int i=0; i<num_ferm; ++i){
        ferm_part_i[i] = bin_state_i[i];
        ferm_part_j[i] = bin_state_j[i];
    }

    if(bin_state_i != bin_state_j){  // This part of H is diagonal.
        return 0.0;
    } else if(bin_to_dec(ferm_part_i, num_ferm)%3==0){
        return 0.0;
    } else if((bin_to_dec(ferm_part_i, num_ferm)-1)%3==0 && bin_to_dec(ferm_part_i, num_ferm) != 10){
        return -mass;
    } else if((bin_to_dec(ferm_part_i, num_ferm)+1)%3==0 && bin_to_dec(ferm_part_i, num_ferm) != 5){
        return mass;
    } else if(bin_to_dec(ferm_part_i, num_ferm) == 10){
        return 2*mass;
    } else{
        return -2*mass;
    }
}

// Apply H_g(i) = sigma_(i(i+1))^{x} on a single link.
void Apply_HgagugeLink_site(vector<int>& bin_state_out, int site){
    bin_state_out[dimstate-site-1] = (bin_state_out[dimstate-site-1]+1)%2;
}


// Compute the element <j|H_g|i> where H_g = \sum_{i=0}^{dimstate} sigma_(i(i+1))^{x}
double H_gauge_link(vector<int>& bin_state_j, vector<int>& bin_state_i){
    vector<int> bin_state_aux(dimstate);

    for(int i=0;i<3;++i){
        for(int j=0;j<dimstate;++j){
            bin_state_aux[j]=bin_state_i[j];
        }
        Apply_HgagugeLink_site(bin_state_aux, i);
        if(bin_state_aux==bin_state_j){
          return 1.0;
        }
    }
    return 0.0;
}


// Apply the operator H_hopX (i) = \frac{1}{4}(-1)^i\sigma^{z}_{(i(i+1))}\sigma^{x}_{i}i\sigma^{x}_{i+1} --- i is the site of the lattice.
void Apply_HhopX_site(vector<int>& state_in, vector<int>& state_out, int site, double *amp){
    state_out[num_ferm-site]   = (state_in[num_ferm-site]+1)%2;
    state_out[num_ferm-site-1] = (state_in[num_ferm-site-1]+1)%2;

    *amp = 0.25*pow(-1, site);
    
    if(state_in[dimstate-site]==1){
       *amp = -*amp;
    }
}


// Compute the element <j|H_hopX|i> where  H_hopX = \sum_{i=0}^{i=dimstate} \frac{1}{4}(-1)^i\sigma^{z}_{(i(i+1))}\sigma^{x}_{i}i\sigma^{x}_{i+1}
double H_hop_X(vector<int>& bin_state_j, vector<int> bin_state_i){
  vector<int> state_aux(dimstate);
      double amp=0;

      for(int site=1; site<4; ++site){
        for(int i=0;i<dimstate;++i){
            state_aux[i]=bin_state_i[i];
        }
        Apply_HhopX_site(bin_state_i, state_aux, site, &amp);
        if(bin_state_j == state_aux){
            return amp;
        }        
      }
      return 0.0;
}

// Apply the operator H_hopY (i) = \frac{1}{4}(-1)^i\sigma^{z}_{(i(i+1))}\sigma^{y}_{i}i\sigma^{y}_{i+1} --- i is the site of the lattidouble Apply_HhopY_site(vector<int>& state_in, vector<int>& state_out, int site, double* amp){
void Apply_HhopY_site(vector<int>& state_in, vector<int>& state_out, int site, double* amp){
    state_out[num_ferm-site] = (state_in[num_ferm-site]+1)%2;
    state_out[num_ferm-site-1] = (state_in[num_ferm-site-1]+1)%2;

    *amp = 0.25*pow(-1, site+1);

    if(state_in[dimstate-site]==1){
        *amp = -*amp;
    }
    if(state_out[num_ferm-site]==0){
        *amp = -*amp;
    }
    if(state_out[num_ferm-site-1]==0){
        *amp = -*amp;
    }
}



// Compute the element <j|H_hopY|i> where  H_hopY = \sum_{i=0}^{i=dimstate} \frac{1}{4}(-1)^i\sigma^{z}_{(i(i+1))}\sigma^{y}_{i}i\sigma^{y}_{i+1}
double H_hop_Y(vector<int>& bin_state_j, vector<int> bin_state_i){
    vector<int> state_aux(dimstate);
    
    double amp=0;

    for(int site=1; site<4; ++site){
      for(int i=0;i<dimstate;++i){
          state_aux[i]=bin_state_i[i];
      }
      Apply_HhopY_site(bin_state_i, state_aux, site, &amp);
      if(bin_state_j == state_aux){
          return amp;
      }        
    }
    return 0.0;
}


     
int main(int argc, char ** argv){
    
    cout<<"\n\nZ2 gauge theory with staggered fermions. 3 links and 4 fermions.\n"<<endl;

    if(argc<5){
        cout<<"Generate H matrix for a Z2 gauge theory plus staggered fermions and compute the evolution of the -number density- of the leftmost fermion. The lattice is 1D and there are 4 staggered fermions and 3 gauge links. The initial state is the one with all 0s projected on the gauge invariant subspace\n\n";
        cout<<"usage:\n\n\t./z2_gauge_matter_solver <mass_parameter> <stepsize> <number of total steps> <name of generated file>"<<endl;
        exit(1);
    }


    double mass = stod(argv[1]);
    double dt = stod(argv[2]);
    int n = stod(argv[3]);
    string filestem = argv[4];

    vector<int> a_state(dimstate);
    vector<int> b_state(dimstate);

    vec eigvals;
    mat eigvecs;

    string eigvalscachename = filestem+"_vals_cached";
    string eigvecscachename = filestem+"_vecs_cached";
    
    if(not ifstream(eigvalscachename.c_str()).good()){
        
/*---------------------Building the Matrix-------------*/

        cout<<"Construct the matrix ..."<<flush;      
        tglob.tic();

        printf("\n");
        
        mat H((int)pow(2,dimstate), (int)pow(2,dimstate));
        for(int a=0;a<(int) pow(2,dimstate);++a){
            for(int b=0;b<(int) pow(2,dimstate);++b){
                dec_to_bin(a, a_state, dimstate);
                dec_to_bin(b, b_state, dimstate);
                H(b,a) = H_mass_ferm(b_state, a_state, mass);
                H(b,a) += H_gauge_link(b_state, a_state);
                H(b,a) += H_hop_X(b_state, a_state);
                H(b,a) += H_hop_Y(b_state, a_state);
//                if(H(b,a) != 0){
//                    printf(" %d, %d -> %.12lf\n", b, a, H(b,a));
//                }
            }   
        }
       
        
        for(int a=0;a<(int) pow(2,dimstate);++a){
            for(int b=0;b<(int) pow(2,dimstate);++b){
                if(H(a,b) != H(b,a)){
                    printf("NOT SYMMETRIC!\n");
                }
            }
        }


        tglob.toc();
        cout<<"\b\b\b"<<DONE " in "<<tglob.toc()<<" seconds."<<endl;

        cout<<"H diagonalization ..."<<flush;
        tglob.tic();

        // The actual diagonalization.
        eig_sym(eigvals, eigvecs, H);

        tglob.toc();
        cout<<"\b\b\b"<<DONE<<" in "<<tglob.toc()<<" seconds."<<endl;

        // cache decomposed H matrix
       
        cout<<eigvals<<"\n"<<endl;

        eigvals.save(eigvalscachename);
        eigvecs.save(eigvecscachename);
    }else{
        cout<<"\033[7;33mUsing cached eigenstuff\033[0m"<<endl;
        eigvals.load(eigvalscachename);
        eigvecs.load(eigvecscachename);
    }    
  
/*-------------- Evolution -----------*/
    cout<<"Evolution ..."<<flush;
    tglob.tic();

    // initialize state according to the paper of Lamm.
    vector<Complex> psi0(Dim,0.0);
    
    psi0[24]={1.0/sqrt(8),0.0};
    psi0[25]={-1.0/sqrt(8),0.0};
    psi0[26]={1.0/sqrt(8),0.0};
    psi0[27]={-1.0/sqrt(8),0.0};
    psi0[28]={1.0/sqrt(8),0.0};
    psi0[29]={-1.0/sqrt(8),0.0};
    psi0[30]={1.0/sqrt(8),0.0};
    psi0[31]={-1.0/sqrt(8),0.0};

    // initialize auxiliary buffer for the evolution and pointers
    vector<Complex> psi0_mom(Dim,0.0);
    
    int i, L, Lp;
    double t = 0.0;
    double num_den_tmp;
    
    vector<double> num_den(n+1,0.0);
    vector<int> bin_state(dimstate);
    
    
    //    #pragma omp parallel for collapse(2)
    for(Lp = 0; Lp < Dim; ++Lp){
        for(L = 0; L < Dim; ++L){
            psi0_mom[Lp] += eigvecs(L,Lp)*psi0[L];
        }
    }    

    // compute number density on the state
    num_den_tmp=0.0;

    #pragma omp parallel for reduction(+:num_den_tmp)
    for(L = 0; L < Dim; ++L){
        dec_to_bin(L, bin_state, dimstate);
        num_den_tmp+=bin_state[3]*norm(psi0[L]);
    }
    num_den[0]=num_den_tmp;

    // iterate solver
    #pragma omp parallel private(t,Lp,L,i,num_den_tmp)
    {
        vector<Complex> psi(psi0);
        vector<Complex> psi_aux(Dim,0.0);

        #pragma omp for 
        for(i = 1; i < n+1; ++i){
            t=i*dt;
            // apply phases
            for(Lp = 0; Lp < Dim; ++Lp){
                psi_aux[Lp]=exp(-iu*eigvals(Lp)*t)*psi0_mom[Lp];
            }

            // transform back to standard basis
            for(L = 0; L < Dim; ++L){
                psi[L]=0.0;
                for(Lp = 0; Lp < Dim; ++Lp){
                    psi[L]+=psi_aux[Lp]*eigvecs(L,Lp);
                }
            }

            // compute plaquette on the state
            num_den_tmp=0.0;
            for(L = 0; L < Dim; ++L){
                dec_to_bin(L, bin_state, dimstate);
                num_den_tmp+=bin_state[3]*norm(psi[L]);
            }
            num_den[i] = num_den_tmp;
       }

    }

    tglob.toc();
    cout<<endl;
    cout<<"\b\b\b"<<DONE<<" in "<<tglob.toc()<<" seconds."<<endl;

    // save data
    FILE * NumDen_file = fopen((filestem+"_NumDen.dat").c_str(),"w");
    for(i = 0; i < n+1; ++i){
        fprintf(NumDen_file,"%.2f %.10lf\n",i*dt,num_den[i]);
    }
    fclose(NumDen_file);
  
    return 0;
}
