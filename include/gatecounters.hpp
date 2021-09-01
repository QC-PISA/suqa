#pragma once
#ifdef GATECOUNT
#include <math.h>
#include <string>
#include <vector>
#include <iostream>

typedef unsigned int uint;

struct GateRecord{
    uint ng1=0;  // 1 qubit gate 
    uint ng2=0;  // 2 qubit gate
    GateRecord(uint gi1=0,uint gi2=0) : ng1(gi1), ng2(gi2) {}
    GateRecord(GateRecord&& gr) : ng1(gr.ng1), ng2(gr.ng2) {}
};
class GateCounter{
public:
    std::string name;
    std::vector<GateRecord> grecords;

    GateCounter(std::string nname) : name(nname) {}

    void increment_g1g2(uint ng1, uint ng2){
        if(active){
            grecords.back().ng1+=ng1;
            grecords.back().ng2+=ng2;
        }
    }
    void increment_g1g2(const GateRecord& gr){
        if(active){
            grecords.back().ng1+=gr.ng1;
            grecords.back().ng2+=gr.ng2;
        }
    }

    void get_info(double& mean1, double &err1, double& mean2, double& err2, size_t& iters) const{
       get_info(mean1,err1,mean2,err2,iters,grecords); 
    }

    inline void activate(){ new_record(); }
    inline void deactivate(){ active=false; }

    // In a naive implementation, an n-control Toffoli can be described
    // by 2*n-3 standard 2-control Toffoli gates + 1 CNOT, while
    // a single 2-control Toffoli can be written using 9 1-qubit + 6 CNOT gates
    // Some smarter implementations probably exist, 
    // but there are just lower bounds (e.g., see https://arxiv.org/pdf/0803.2316.pdf)
    // We also assume that both types of control (set and unset) are equally available.
    static GateRecord n_ctrl_toffoli_gates(uint n){
        GateRecord gr;
        if(n==0){
            gr.ng1=1; 
        }else if(n==1){
            gr.ng2=1; 
        }else if(n==2){
            gr.ng1=9; 
            gr.ng2=6; 
        }else{
            gr.ng1=9*(2*n-3); 
            gr.ng2=6*(2*n-3)+1; 
        }
        return gr;
    }

    void new_record(){
        active = true;
        grecords.push_back(GateRecord());
    }

    void clear(){
        active = false; 
        grecords.clear();
    }

private:

    // assuming independent samplings
    void get_info(double& mean1, double &err1, double& mean2, double &err2, size_t &iters, const std::vector<GateRecord>& grecords) const{ 
        iters=grecords.size();
        mean1 = 0.0;
        mean2 = 0.0;
        err1 = 0.0;
        err2 = 0.0;
        for(const auto& el : grecords){
            mean1 +=el.ng1;
            mean2 +=el.ng2;
            err1 +=el.ng1*el.ng1;
            err2 +=el.ng2*el.ng2;
        }
        if(iters>0){
            mean1 /= (double)iters;
            mean2 /= (double)iters;
        }
        if(iters>1){
            err1 = sqrt(abs(err1/(double)iters - mean1*mean1)/(iters-1.0));
            err2 = sqrt(abs(err2/(double)iters - mean2*mean2)/(iters-1.0));
            if(err1<1e-10) err1=0.0;
            if(err2<1e-10) err2=0.0;
        }else{
            err1=err2=0;
        }
    }

    bool active=false; // unactive by default
};

class GateCounterList{
public:
    uint gc_mask_set_qbits=0;
    std::vector<GateCounter*> counters;

    void add_counter(GateCounter* gctr){
        counters.push_back(gctr);
    }

    void increment_g1g2(uint ng1, uint ng2){
        for(GateCounter* gc : counters){
            gc->increment_g1g2(ng1,ng2);
        }
    }

    void increment_g1g2(const GateRecord& gr){
        for(GateCounter* gc : counters){
            gc->increment_g1g2(gr.ng1,gr.ng2);
        }
    }

    void update_cmask_setbits(uint cmask){
        uint count=0, n=cmask;
        while(n!=0){
            if((n & 1) == 1)
                count++;
            n>>=1;
        }
        gc_mask_set_qbits=count;
    }

    void print_counts(){
        if(counters.empty()) return;
        printf("\n\nGate counts\n");
        size_t samples;
        double m1, e1, m2, e2;
        printf("\n%-15s  %-9s  %-9s  %-9s  %-9s  %-9s  %-9s  %-9s\n","counter name","records","tot_ng1","<ng1>","d<ng1>","tot_ng2","<ng2>","<dng2>");
        for(const GateCounter* gc : counters){
            gc->get_info(m1,e1,m2,e2,samples);
            printf("%-15s  %-9zu  %-9u  %-9.3lg  %-9.3lg  %-9u  %-9.3lg  %-9.3lg\n",gc->name.c_str(),samples,(uint)(round(m1*samples)),m1,e1,(uint)(round(m2*samples)),m2,e2);
        }
        printf("\n");
    }

    void print_counts_detailed(){
        if(counters.empty()) return;
        printf("\n\nDetailed gate counts\n");
        for(const GateCounter* gc : counters){
            printf("\ngate counter '%s' (%zu records):\n",gc->name.c_str(),gc->grecords.size());
            uint rec_ctr=0;
            for(const GateRecord& gr : gc->grecords){
                printf("record %u: ng1=%u, ng2=%u\n",rec_ctr++,gr.ng1,gr.ng2);
            }
        }
        printf("\n");
    }
};
#endif
