#pragma once
#ifdef GATECOUNT
#include <math.h>
#include <vector>

typedef unsigned int uint;

struct GateRecord{
    uint ng1=0;  // 1 qubit gate 
    uint ng2=0;  // 2 qubit gate
    GateRecord(uint gi1=0,uint gi2=0) : ng1(gi1), ng2(gi2) {}
    GateRecord(const GateRecord& gr) : ng1(gr.ng1), ng2(gr.ng2) {}
};
struct GateCounter{
public:
    std::vector<GateRecord> gs;

    void increment_g1g2(uint ng1, uint ng2){
        if(active){
            gs.back().ng1+=ng1;
            gs.back().ng2+=ng2;
        }
    }
    void increment_g1g2(const GateRecord& gr){
        if(active){
            gs.back().ng1+=gr.ng1;
            gs.back().ng2+=gr.ng2;
        }
    }

    void get_meanstd(double& mean1, double &err1, double& mean2, double& err2){
       get_meanstd(mean1,err1,mean2,err2,gs); 
    }

    inline void activate(){ new_record(); active=true; }
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
        gs.push_back(GateRecord());
    }
private:

    // assuming independent samplings
    void get_meanstd(double& mean1, double &err1, double& mean2, double &err2, const std::vector<GateRecord>& gs){
        mean1 = 0.0;
        mean2 = 0.0;
        err1 = 0.0;
        err2 = 0.0;
        for(const auto& el : gs){
            mean1 +=el.ng1;
            mean2 +=el.ng2;
            err1 +=el.ng1*el.ng1;
            err2 +=el.ng2*el.ng2;
        }
        if(gs.size()>0){
            mean1 /= (double)gs.size();
            mean2 /= (double)gs.size();
        }
        if(gs.size()>1){
            err1 = sqrt((err1/(double)gs.size() - mean1*mean1)/(gs.size()-1.0));
            err2 = sqrt((err2/(double)gs.size() - mean2*mean2)/(gs.size()-1.0));
        }
    }

    bool active=false; // unactive by default
};

struct GateCounterList{
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

};
#endif
