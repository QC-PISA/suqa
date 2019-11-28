#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

using namespace std;

struct arg_list{
    double beta = 0.0;
    double h = 0.0;
    int metro_steps = 0;
    int reset_each = 0;
    int ene_qbits = 0;
    string outfile = "";
    int max_reverse_attempts = 100;
    unsigned long long int seed = 0;
    double pe_time_factor = 1.0;
    int pe_steps = 10; 
    string Xmatstem = "";
    
    friend ostream& operator<<(ostream& o, const arg_list& al);
};

ostream& operator<<(ostream& o, const arg_list& al){
    o<<"beta: "<<al.beta<<endl;
    o<<"h: "<<al.h<<endl;
    o<<"metro steps: "<<al.metro_steps<<endl;
    o<<"reset each: "<<al.reset_each<<endl;
    o<<"num E qbits "<<al.ene_qbits<<endl;
    o<<"max reverse attempts: "<<al.max_reverse_attempts<<endl;
    o<<"seed: "<<al.seed<<endl;
    o<<"out datafile: "<<al.outfile<<endl;
    o<<"time of PE evolution: "<<al.pe_time_factor<<endl;
    o<<"steps of PE evolution: "<<al.pe_steps<<endl;
    o<<"file stem for X measure: "<<al.Xmatstem<<endl;
    return o;
}

void parse_arguments(arg_list& args, int argc, char** argv){
    int fixed_args = 6;
    map<string,int> argmap;
    map<int,string> argmap_inv;
    char *end;
    int base_strtoull = 10;

    // fixed arguments
    args.beta = stod(argv[1],NULL);
    args.h = stod(argv[2],NULL);
    args.metro_steps = atoi(argv[3]);
    args.reset_each = atoi(argv[4]);
    args.ene_qbits = atoi(argv[5]);
    args.outfile = argv[6];

    // floating arguments
    for(int i = fixed_args+1; i < argc; ++i){
        argmap[argv[i]]=i;
        argmap_inv[i]=argv[i];
    }
    int tmp_idx;

    // (int) max_reverse_attempts
    tmp_idx = argmap["--max-reverse"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--max-reverse' flag"; 
       
       args.max_reverse_attempts = atoi(argmap_inv[tmp_idx+1].c_str()); 
    }

    // (unsigned long long) seed
    tmp_idx = argmap["--seed"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--seed' flag"; 
       
       args.seed = strtoull(argmap_inv[tmp_idx+1].c_str(), &end, base_strtoull); 
    }

    // (double) pe_time_factor
    tmp_idx = argmap["--PE-time"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--PE-time' flag"; 
       
       args.pe_time_factor *= stod(argmap_inv[tmp_idx+1].c_str(), NULL); 
    }

    // (int) pe_steps
    tmp_idx = argmap["--PE-steps"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--PE-steps' flag"; 
       
       args.pe_steps = stod(argmap_inv[tmp_idx+1].c_str(), NULL); 
    }

    // (int) pe_steps
    tmp_idx = argmap["--X-mat-stem"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--X-mat-stem' flag"; 
       
       args.Xmatstem = argmap_inv[tmp_idx+1]; 
    }

    // argument checking
    if(args.beta <= 0.0){
        throw "ERROR: argument <beta> invalid";
    }

    if(args.metro_steps <= 0){
        throw "ERROR: argument <metro steps> invalid";
    }

    if(args.reset_each <=0){
        throw "ERROR: argument <reset each> non positive";
    }
    
    if(args.outfile == ""){
        throw "ERROR: argument <output file path> empty";
    }

    if(args.max_reverse_attempts <=0){
        throw "ERROR: argument <max reverse attempts> non positive";
    }

    if(args.pe_time_factor <=0){
        throw "ERROR: argument <time of PE evolution> non positive";
    }

    if(args.pe_steps <=0){
        throw "ERROR: argument <steps of PE evolution> non positive";
    }
}
