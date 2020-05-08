#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <cstring>
#include <stdio.h>
#include <cmath>
#include <cassert>
#include "Rand.hpp"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include "io.hpp"
#include "parser.hpp"
#include "suqa.cuh"
#include "system.cuh"
#include "qms.cuh"

using namespace std;





#define NUM_THREADS 128
#define MAXBLOCKS 65535
uint suqa::threads;
uint suqa::blocks;
cudaStream_t suqa::stream1, suqa::stream2;

double *host_state_re_lollo, *host_state_im_lollo; 


void sparse_print(double *v_re, double *v_im, uint size, FILE *file){
  // for non-contiguous even-odd entries corresponding to real and imag parts
  size_t index_size = (int)std::round(std::log2(size));
  for(uint i=0; i<size; ++i){
    std::complex<double> var(v_re[i],v_im[i]);
       if(norm(var)>1e-10){
	 std::string index= std::bitset<32>(i).to_string();
	 index.erase(0,32-index_size);
	 fprintf(file,"%i -> (%.3e, %.3e)\n",i,norm(var),atan2(v_im[i],v_re[i]));
       }
    }
    fprintf(file,"STOP\n");
}

void setup_PE_test(){
    qms::fill_rphase(qms::ene_qbits+1);
    //    qms::fill_bitmap();
    qms::bm_states.resize(qms::state_qbits);
    qms::bm_enes_old.resize(qms::ene_qbits);
    uint c=0;
    for(uint i=0; i< qms::state_qbits; ++i)  qms::bm_states[i] = c++;
    for(uint i=0; i< qms::ene_qbits; ++i)    qms::bm_enes_old[i] = c++;

#if !defined(NDEBUG)
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_re,qms::Dim*sizeof(double),cudaHostAllocDefault));
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_im,qms::Dim*sizeof(double),cudaHostAllocDefault));
#endif
}


// simulation parameters
double beta;
double h;
int thermalization;

// defined in src/system.cu
void init_state(ComplexVec& state, uint Dim, uint j=0);

arg_list args;

void save_measures(string outfilename){
    FILE * fil = fopen(outfilename.c_str(), "a");
    for(uint ei = 0; ei < qms::E_measures.size(); ++ei){
        fprintf(fil, "%.16lg %.16lg\n", qms::E_measures[ei], qms::X_measures[ei]);
    }
    fclose(fil);
    qms::E_measures.clear();
    qms::X_measures.clear();
}

void deallocate_state(ComplexVec& state){
    if(state.data!=nullptr){
        HANDLE_CUDACALL(cudaFree(state.data));
    }
    state.vecsize=0U;
}

void allocate_state(ComplexVec& state, uint Dim){
    if(state.data!=nullptr or Dim!=state.vecsize)
        deallocate_state(state);


    state.vecsize = Dim; 
    HANDLE_CUDACALL(cudaMalloc((void**)&(state.data), 2*state.vecsize*sizeof(double)));
    // allocate both using re as offset, and im as access pointer.
    state.data_re = state.data;
    state.data_im = state.data_re + state.vecsize;
}

void print_state(ComplexVec& state, FILE* file){
  // extern double *host_state_re_lollo, *host_state_im_lollo; 
  cudaMemcpyAsync(host_state_re_lollo,state.data_re,state.size()*sizeof(double),cudaMemcpyDeviceToHost,suqa::stream1);
  cudaMemcpyAsync(host_state_im_lollo,state.data_im,state.size()*sizeof(double),cudaMemcpyDeviceToHost,suqa::stream2);
  cudaDeviceSynchronize();

  sparse_print(host_state_re_lollo,host_state_im_lollo,state.size(),file);
}



void quantum_phase_tour(int nsteps, FILE* file){
  // qms::reset_non_state_qbits(qms::gState);
  std::vector<uint> c_E_olds(qms::ene_qbits,0);
  for(int i=0; i<nsteps; i++){
    for(uint ei=0; ei<qms::ene_qbits; ei++)
      suqa::apply_reset(qms::gState,qms::bm_enes_old[ei],qms::rangen.doub());
    qms::apply_Phi_old();
  }

  suqa::measure_qbits(qms::gState, qms::bm_enes_old, c_E_olds, qms::extract_rands(qms::ene_qbits));
  double tmp_E=qms::t_PE_shift+qms::creg_to_uint(c_E_olds)/(double)(qms::t_PE_factor*qms::ene_levels);
  cout<<tmp_E<<endl;
  
  print_state(qms::gState, file);
  
  for(uint ei=0; ei<qms::ene_qbits; ei++)
    suqa::apply_reset(qms::gState,qms::bm_enes_old[ei],qms::rangen.doub());
  
  
  std::cout<<"fatto!"<<std::endl;
}


void print_isto(vector<pair<long double,long double>> &isto, FILE *output){
  long double sum=0;
  for(auto &a:isto)fprintf(output,"%Lf %Lf\n",a.first,a.second);
  fprintf(output,"\n\n");
  // for(auto a:isto)cout<<a.first<<" "<<a.second<<endl;
  for(auto &a:isto)sum+=a.second;
  cout<<"This number should be 1: "<<sum<<endl;
}


void ciao(){cout<<"ciao"<<endl;}
void niente(){cout<<" "<<endl;}

void moves_spectrum(FILE* phile,vector<pair<long double,long double>> & saved_isto){
  vector<pair<long double,long double>> ret;
  const double E_min=args.ene_min, E_max=args.ene_max;
  double probE1=0, E=E_min;
  double dE=(E_max-E_min)/(double)(qms::ene_levels-1);
  double bincount=0.1;
  uint i=0;
  vector<pair<long double,long double>> tempstate;
  long double square_mod,phase;
  char c[6]="";
  uint tempi=1;

  while(fscanf(phile,"%i -> (%Le, %Le)\n",&i,&square_mod,&phase)!=EOF){
    int a=i/qms::state_levels-tempi/qms::state_levels;
    E+=(a*dE);
    if(i==tempi // and bincount!=0.1
       ){
      if(E<E_max+dE){
	E+=dE;
	a+=1;
      }
      else{
      	ret.push_back(make_pair(E-dE,probE1));
      	probE1=0;
      }
    }
    if(a!=0){
      ret.push_back(make_pair(E-dE,probE1));
      probE1=0;
      bincount+=1.0;
    }
    if (i!=tempi)
      probE1+=square_mod;
    
    if(E>E_max+(1.5*dE)){
      cout<<i<<" "<<tempi<<" "<<E<<endl;
      cout<<"Problem in energy binning: E > "<<E_max<<endl;
      exit(1);
    }
    if(tempi==i and bincount!=0.1){
      cout<<fgets(c,6,phile)<<endl;
      if(ret.size()==0) ret.push_back(make_pair(E_min,probE1));
      
      if(saved_isto.size()==ret.size())
	for(uint i=0;i<saved_isto.size();i++)
	  saved_isto[i].second+=ret[i].second;

      else if(saved_isto.size()==0)saved_isto=ret;
      else if(saved_isto.size()!=saved_isto.size()){cout<<"I don't know how, but it happened"<<endl;exit(1);}
      ciao();
      return niente();
    }
    tempi=i;
  }
  cout<<"some serious error in loop"<<endl;exit(1);
}


void phase_spectrum(FILE* phile, FILE *output){
  vector<pair<long double,long double>> ret;
  const double E_min=args.ene_min, E_max=args.ene_max;
  double probE1=0, E=E_min;
  double dE=(E_max-E_min)/(double)(qms::ene_levels-1);
  double bincount=0.1;
  uint i=0;
  vector<pair<double,double>> tempstate;
  double square_mod,phase;
  char c[6]="";
  uint tempi=1;
  
  while(fscanf(phile,"%i -> (%le, %le)\n",&i,&square_mod,&phase)!=EOF){
    
    int a=i/qms::state_levels-tempi/qms::state_levels;
    E+=(a*dE);

    if(i==tempi and bincount!=0.1){
      if(E<E_max+dE){
	E+=dE;
	a+=1;
      }
      else {
      	ret.push_back(make_pair(E-dE,probE1));
      	probE1=0;
      }
    }

    if(a!=0){
      ret.push_back(make_pair(E-dE,probE1));
      probE1=0;
      bincount+=1.0;
    }

    if (i!=tempi)
      probE1+=square_mod;    

    if(E>E_max+(1.5*dE)){
      cout<<i<<" "<<E<<endl;
      cout<<"Problem in energy binning: E > "<<E_max<<endl;
      exit(1);
    }
    
    if(tempi==i and bincount!=0.1){
      cout<<fgets(c,6,phile)<<endl;
      if(ret.size()==0) ret.push_back(make_pair(E_min,probE1));
      return print_isto(ret,output);
    }
    tempi=i;
  }
  cout<<"some serious error in loop"<<endl;exit(1);
}



int main(int argc, char** argv){
    if(argc < 4){
        printf("usage: %s <g_beta> <num state qbits> <num ene qbits> [--seed <seed> (random)] [--ene-min <min energy> (0.0)] [--ene-max <max energy> (1.0)] [--PE-steps <steps of PE evolution> (10)] \n", argv[0]);
        exit(1);
    }

    parse_arguments_PE_test(args, argc, argv);

    g_beta = args.g_beta; // defined as extern in system.cuh
    qms::state_qbits = (uint)args.state_qbits;
    qms::ene_qbits = (uint)args.ene_qbits;
    string outfilename(args.outfile);
    qms::max_reverse_attempts = (uint)args.max_reverse_attempts;
    qms::n_phase_estimation = args.pe_steps;
    qms::iseed = args.seed;
    if(qms::iseed>0)
        qms::rangen.set_seed(qms::iseed);
    
    qms::iseed = qms::rangen.get_seed();

    qms::nqubits = qms::state_qbits + qms::ene_qbits;
    qms::Dim = (1U << qms::nqubits);
    qms::ene_levels = (1U << qms::ene_qbits);
    qms::state_levels = (1U << qms::state_qbits);

    qms::t_PE_shift = args.ene_min;
    qms::t_PE_factor = (qms::ene_levels-1)/(double)(qms::ene_levels*(args.ene_max-args.ene_min)); 
    qms::t_phase_estimation = qms::t_PE_factor*8.*atan(1.0);

    suqa::threads = NUM_THREADS;
    suqa::blocks = (qms::Dim+suqa::threads-1)/suqa::threads;
    if(suqa::blocks>MAXBLOCKS) suqa::blocks=MAXBLOCKS;

    
    // Banner
    suqa::print_banner();
    cout<<"arguments:\n"<<args<<endl;

    // Initialization of utilities. There are not new energy nor acceptance registers.
    suqa::setup(qms::Dim);
    setup_PE_test();
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_re_lollo,qms::Dim*sizeof(double),cudaHostAllocDefault));
    HANDLE_CUDACALL(cudaHostAlloc((void**)&host_state_im_lollo,qms::Dim*sizeof(double),cudaHostAllocDefault));
    // Initialization:
    // known eigenstate of the system (see src/system.cu)
    
    allocate_state(qms::gState, qms::Dim);

    vector<vector<pair<long double,long double>>> super_duper_isto(200);    

    uint inizio_mossa=0;
    uint nmosse=13;
    int nstati=1;
    
    double EV[8]={5.8557980376081717,5.9073497011647422,6.6746335458672430,6.7261852094238153,9.1073497011647433,9.1589013647213164,9.9261852094238154,9.9777368729803815};
    double dE=(args.ene_max-args.ene_min)/(double)(qms::ene_levels-1);
      // vector <double> grid(qms::ene_levels);
      // for(uint i=0;i<grid.size();i++) grid[i]=args.ene_min+dE*i;
      // for(auto &a:EV){
      // 	int count=0;
      // 	while(grid[count]<a)
      // 	  count++;
      // 	a=grid[count]-dE;
      // }
      
      for(int j=0;j<nstati;j+=1){
	init_state(qms::gState,qms::Dim,j);
	vector<pair<long double,long double>> super_isto;

	
      FILE *debug;
      debug=fopen("debug","a");
      print_state(qms::gState, debug);
      fclose(debug);

      FILE *temp;

      FILE *output;
      FILE *plottatore;

      char name[1000];
      sprintf(name,"enequbit%i_range%f-%f_ev%i",qms::ene_qbits,args.ene_min,args.ene_max,j);
      output=fopen(name,"a");
      char plottaname[1011];
      sprintf(plottaname,"plotta_%s",name);
      plottatore=fopen(plottaname,"a");

      /*******************************SOLO GIRI DI PHASE ESTIMATION****************************/
      /*
      temp=fopen("temp","a");
      quantum_phase_tour(1,temp);
      fclose(temp);

      FILE *debug;
      debug=fopen("debug","a");
      print_state(qms::gState, debug);
      fclose(debug);

      
      temp=fopen("temp","r");	
      phase_spectrum(temp,output);
      fprintf(plottatore,"set title \"QPE on gauge invariant base vector %d \" font \",22\"\n",j);
      fprintf(plottatore,"set xlabel \"E\" font \",18\"\n");
      fprintf(plottatore,"set ylabel \"P(E)\" font \",18\"\n");
      
      fprintf(plottatore,"set box %lf absolute\n",dE);
      fprintf(plottatore,"set style fill solid 0.5\n");
      fprintf(plottatore,"$expected_ev << EOD\n");
      for(auto &a : EV) fprintf(plottatore,"%f 1 %lf\n",a,dE);
      fprintf(plottatore,"EOD\n");
      fprintf(plottatore,"set parametric\nset trange[0:1]\n");
      fprintf(plottatore,"plot \"enequbit%i_range%f-%f_ev%i\" using 1:2 w boxes title \"initial state |%d>\" ,\\\n",qms::ene_qbits,args.ene_min,args.ene_max,j,j);
      fprintf(plottatore,"$expected_ev w boxes fs transparent solid 0.1 title \"expected energies\" ,\\\n");
      fprintf(plottatore,"%f,t linetype -1 title \"\", %f,t linetype -1 title \"\"\n",args.ene_min,args.ene_max);
      fcloseall();
      
      if(remove( "temp" )!=0) perror( "Error deleting file" );
      else puts( "File successfully deleted" );
    }
      suqa::clear();
      qms::clear();
      HANDLE_CUDACALL(cudaFreeHost(host_state_re_lollo));
      HANDLE_CUDACALL(cudaFreeHost(host_state_im_lollo));
      return 0;
}
      */
      /************************************************************************************/
      temp=fopen("temp","a");
      quantum_phase_tour(1,temp);
      fclose(temp);
      if( remove( "temp" ) != 0 )
	perror( "Error deleting file" );
      else
	puts( "File successfully deleted" );
      
      
      for(uint Ci=inizio_mossa;Ci<inizio_mossa+nmosse;Ci++){
	cout<<Ci<<endl;
	apply_C(qms::gState, Ci);
	
	temp=fopen("temp","a");
	quantum_phase_tour(1,temp);
	fclose(temp);
	
	temp=fopen("temp","r");	
	moves_spectrum(temp, super_isto);
	fclose(temp);
	
	temp=fopen("temp","r");
	moves_spectrum(temp, super_duper_isto[Ci]);
	fclose(temp);
	
       apply_C_inverse(qms::gState, Ci);
	if( remove( "temp" ) != 0 )
	  perror( "Error deleting file" );
	else
	  puts( "File successfully deleted" );
      }
      for(auto &a:super_isto)
	a.second/=(long double)nmosse;
      
      print_isto(super_isto,output);
      fprintf(plottatore,"set title \"QPE on omni-moved eigenvector %d \" font \",22\"\n",j);
      fprintf(plottatore,"set xlabel \"E\" font \",18\"\n");
      fprintf(plottatore,"set ylabel \"P(E)\" font \",18\"\n");
      
      fprintf(plottatore,"set box %lf absolute\n",dE);
      fprintf(plottatore,"set style fill solid 0.5\n");
      fprintf(plottatore,"$expected_ev << EOD\n");
      for(auto &a : EV) fprintf(plottatore,"%f 1 %lf\n",a,dE);
      fprintf(plottatore,"EOD\n");
      fprintf(plottatore,"set parametric\nset trange[0:1]\n");
      
      fprintf(plottatore,"plot \"enequbit%i_range%f-%f_ev%i\" using 1:2 w boxes title \"initial state |%d>\" ,\\\n",qms::ene_qbits,args.ene_min,args.ene_max,j,j);
      
      fprintf(plottatore,"$expected_ev w boxes fs transparent solid 0.1 title \"expected energies\" ,\\\n");
      fprintf(plottatore,"%f,t linetype -1 title \"\", %f,t linetype -1 title \"\"\n",args.ene_min,args.ene_max);
      
      
      fcloseall();
      super_isto.clear();
      }
      
    for(uint Ci=inizio_mossa;Ci<inizio_mossa+nmosse;Ci++){
      char name[1000];
      FILE *output2, *plottatore2;
      sprintf(name,"SUPER_DUPER_enequbit%i_range%f-%f_MOSSA%i",qms::ene_qbits,args.ene_min,args.ene_max,Ci);
      output2=fopen(name,"a");
      char plottaname[1011];
      sprintf(plottaname,"plotta_%s",name);
      plottatore2=fopen(plottaname,"a");
      
      for(auto &a:super_duper_isto[Ci])
	a.second/=(long double)nstati;

      print_isto(super_duper_isto[Ci],output2);
      fprintf(plottatore2,"set title \"Application of move  %d on every eigenstate \" font \",22\"\n",Ci);
      fprintf(plottatore2,"set xlabel \"E\" font \",18\"\n");
      fprintf(plottatore2,"set ylabel \"P(E)\" font \",18\"\n");
      
      fprintf(plottatore2,"set box %lf absolute\n",dE);
      fprintf(plottatore2,"set style fill solid 0.5\n");
      fprintf(plottatore2,"$expected_ev << EOD\n");
      for(auto &a : EV) fprintf(plottatore2,"%f 1 %lf\n",a,dE);
      fprintf(plottatore2,"EOD\n");
      fprintf(plottatore2,"set parametric\nset trange[0:1]\n");
      
      
      fprintf(plottatore2,"plot $expected_ev w boxes fs transparent solid 0.1 title \"expected energies\" ,\\\n");      
      fprintf(plottatore2,"\"SUPER_DUPER_enequbit%i_range%f-%f_MOSSA%i\" using 1:2 w boxes title \"move %d\" ,\\\n",qms::ene_qbits,args.ene_min,args.ene_max,Ci,Ci);
      fprintf(plottatore2,"%f,t linetype -1 title \"\", %f,t linetype -1 title \"\"\n",args.ene_min,args.ene_max);

      fcloseall();
      	


      }
      
      return 0;
}
