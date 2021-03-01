#include "io.hpp"
#include <utility>
#include <vector>
#include <algorithm>

int get_time(struct timeval* tp, struct timezone* tzp){
	(void)tzp;
	namespace sc = std::chrono;
	sc::system_clock::duration d = sc::system_clock::now().time_since_epoch();
	sc::seconds s = sc::duration_cast<sc::seconds>(d);
	tp->tv_sec = s.count();
	tp->tv_usec = sc::duration_cast<sc::microseconds>(d - s).count();

	return 0;
}

bool file_exists(const char* fname){
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__) && !defined(__MINGW32__) && !defined(__MINGW64__)
    return PathFileExistsA(fname);
#else
    return access(fname, F_OK) != -1;
#endif
}

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
    size_t index_size = (int)std::round(std::log2(size));
    for(uint i=0; i<size; ++i){
       std::complex<double> var(v_re[i],v_im[i]);
       if(norm(var)>1e-10){
            std::string index= std::bitset<32>(i).to_string();
            index.erase(0,32-index_size);
//            printf("|%s> (%5d) -> (%.3e, %.3e ) : mod2= %.3e, phase= %.3e pi\n",index.c_str(),i, v_re[i], v_im[i], norm(var),atan2(v_im[i],v_re[i])/M_PI);
            printf("|%s> (%5d) -> sqr ampl: %.3e\n",index.c_str(),i, norm(var));
       }
    }
    std::cout<<std::endl;
}

void qoxo_print(double *v_re, double *v_im, uint size){
    // for non-contiguous even-odd entries corresponding to real and imag parts
    using  uintdoub = std::pair<uint,double>;
    std::vector<uintdoub> state_norm;
    for(uint i=0; i<size; ++i){
       std::complex<double> var(v_re[i],v_im[i]);
       if(norm(var)>1e-8){
           state_norm.push_back(std::make_pair(i,norm(var)));
       }
    }
    std::sort(state_norm.begin(),state_norm.end(),[](const uintdoub &a, const uintdoub &b){return a.second > b.second;});

//    size_t index_size = (int)std::round(std::log2(size));
    printf("games\t\tprobabilities");
    for(const auto& uid : state_norm){
        uint game_stidx = uid.first;
//        std::string index= std::bitset<32>(uid.first).to_string();
//        index.erase(0,32-index_size);
        std::vector<int> gms(9,0);    
        for(uint i=0U; i<9U; ++i){
            gms[i] = (game_stidx & 3U); // 0 -> empty, 1-> first pl, 2-> second pl, 3-> inactive
            game_stidx>>=2U;
        }
        printf("\n%d|%d|%d\n______\n%d|%d|%d\t\t%.3e\n______\n%d|%d|%d\n",gms[0],gms[1],gms[2],gms[3],gms[4],gms[5],uid.second,gms[6],gms[7],gms[8]);
    }
    std::cout<<std::endl;
}

template<class T>
void print(std::vector<T> v){
    for(const auto& el : v)
        std::cout<<el<<" ";
    std::cout<<std::endl;
}

