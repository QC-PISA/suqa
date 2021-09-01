#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

using namespace std;

struct arg_list{
    double beta = 0.25;
    int ene_qbits = 0;
    int szegedy_qbits=0;
    string outfile = "";
    unsigned long long int seed = 0;
    double ene_min = -4.0;
    double ene_max = 8.0;
    double ene_min_szegedy = -1.0;
    double ene_max_szegedy = 3.0;
    int pe_steps = 1;
//    int lambda1_iterations=1;
//    int szegedy_iterations= 1;
    int annealing_sequences=100;
    int sampling=0;

    friend ostream& operator<<(ostream& o, const arg_list& al);
};

ostream& operator<<(ostream& o, const arg_list& al){
    o<<"beta: "<<al.beta<<endl;
    o<<"num E qbits: "<<al.ene_qbits<<endl;
    o<<"num Szegedy qbits: "<<al.szegedy_qbits<<endl;
    o<<"seed: "<<al.seed<<endl;
    o<<"out datafile: "<<al.outfile<<endl;
    o<<"min energy: "<<al.ene_min<<endl;
    o<<"max energy: "<<al.ene_max<<endl;
    o<<"min energy szegedy: "<<al.ene_min_szegedy<<endl;
    o<<"max energy szegedy: "<<al.ene_max_szegedy<<endl;
    o<<"steps of PE evolution: "<<al.pe_steps<<endl;
//    o<<"lambda1_iterations: "<<al.lambda1_iterations<<endl;
//    o<<"szegedy_iterations: "<<al.szegedy_iterations<<endl;
    o<<"annealing_sequences: "<<al.annealing_sequences<<endl;
    o<<"sampling: "<<al.sampling<<endl;
    return o;
}

void parse_arguments(arg_list& args, int argc, char** argv){
    int fixed_args = 5;
    map<string,int> argmap;
    map<int,string> argmap_inv;
    char *end;
    int base_strtoull = 10;

    // fixed arguments
    args.beta = stod(argv[1],NULL);
    args.sampling = atoi(argv[2]);
    args.ene_qbits = atoi(argv[3]);
    args.szegedy_qbits= atoi(argv[4]);
    args.outfile = argv[5];

    // floating arguments
    for(int i = fixed_args+1; i < argc; ++i){
        argmap[argv[i]]=i;
        argmap_inv[i]=argv[i];
    }
    int tmp_idx;

    // flagged arguments without value



    // (unsigned long long) seed
    tmp_idx = argmap["--seed"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--seed' flag";

       args.seed = strtoull(argmap_inv[tmp_idx+1].c_str(), &end, base_strtoull);
    }

    // (double) ene_min
    tmp_idx = argmap["--ene-min"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--ene-min' flag";

       args.ene_min = stod(argmap_inv[tmp_idx+1].c_str(), NULL);
    }

    // (double) ene_max
    tmp_idx = argmap["--ene-max"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--ene-max' flag";

       args.ene_max = stod(argmap_inv[tmp_idx+1].c_str(), NULL);
    }
    // (double) ene_min_szegedy
    tmp_idx = argmap["--ene-min-szegedy"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--ene-min-szegedy' flag";

       args.ene_min_szegedy = stod(argmap_inv[tmp_idx+1].c_str(), NULL);
    }

    // (double) ene_max_szegedy
    tmp_idx = argmap["--ene-max-szegedy"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--ene-max-szegedy' flag";

       args.ene_max_szegedy = stod(argmap_inv[tmp_idx+1].c_str(), NULL);
    }

    // (int) pe_steps
    tmp_idx = argmap["--PE-steps"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--PE-steps' flag";

       args.pe_steps = stod(argmap_inv[tmp_idx+1].c_str(), NULL);
    }




//    // (int) lambda1_iterations
//    tmp_idx = argmap["--lambda1-iterations"];
//    if(tmp_idx>=fixed_args){
//       if(tmp_idx+1>= argc)
//           throw "ERROR: set value after '--lambda1-iterations' flag";
//
//       args.lambda1_iterations = stod(argmap_inv[tmp_idx+1].c_str(), NULL);
//    }
//
//    // (int) szegedy_iterations
//    tmp_idx = argmap["--szegedy-iterations"];
//    if(tmp_idx>=fixed_args){
//       if(tmp_idx+1>= argc)
//           throw "ERROR: set value after '--szegedy-iterations' flag";
//
//       args.szegedy_iterations= stod(argmap_inv[tmp_idx+1].c_str(), NULL);
//    }
    // (int) annealing_sequences
    tmp_idx = argmap["--annealing-sequences"];
    if(tmp_idx>=fixed_args){
       if(tmp_idx+1>= argc)
           throw "ERROR: set value after '--annealing-sequences' flag";

       args.annealing_sequences= stod(argmap_inv[tmp_idx+1].c_str(), NULL);
    }


    // argument checking
    if(args.beta <= 0.0){
        throw "ERROR: argument <beta> invalid";
    }


    if(args.ene_qbits <=0){
        throw "ERROR: argument <num ene qbits> non positive";
    }
    if(args.szegedy_qbits <=0){
        throw "ERROR: argument <num ene qbits> non positive";
    }

    if(args.outfile == ""){
        throw "ERROR: argument <output file path> empty";
    }

    if(args.pe_steps <=0){
        throw "ERROR: argument <steps of PE evolution> non positive";
    }
//    if(args.lambda1_iterations <=0){
//        throw "ERROR: argument <lambda1_iterations> non positive";
//    }
//    if(args.szegedy_iterations <=0){
//        throw "ERROR: argument <szegedy_iterations> non positive";
//    }
    if(args.annealing_sequences <=0){
        throw "ERROR: argument <annealing_sequences> non positive";
    }
    if(args.sampling<=0){
        throw "ERROR: argument <sampling> non positive";
    }
}
