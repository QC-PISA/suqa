#include "io.hpp"
#include <utility>
#include <vector>
#include <algorithm>
#include <sys/ioctl.h> //ioctl() and TIOCGWINSZ
#include <unistd.h> // for STDOUT_FILENO


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

void sparse_print(double *v, size_t size){
    // for contiguous even-odd entries corresponding to real and imag parts
    for(uint i=0; i<size; ++i){
        std::complex<double> var(v[i*2],v[i*2+1]);
        if(norm(var)>1e-10)
            std::cout<<"i="<<i<<" -> "<<var<<"; ";
    }
    std::cout<<std::endl;
}

void sparse_print(double *v_re, double *v_im, size_t size){
    // for non-contiguous even-odd entries corresponding to real and imag parts
    size_t index_size = (int)std::round(std::log2(size));
    for(uint i=0; i<size; ++i){
       std::complex<double> var(v_re[i],v_im[i]);
       if(norm(var)>1e-10){
            std::string index= std::bitset<32>(i).to_string();
            index.erase(0,32-index_size);
            printf("|%s> (%5d) -> (%.3e, %.3e ) : mod2= %.3e, phase= %.3e pi\n",index.c_str(),i, v_re[i], v_im[i], norm(var),atan2(v_im[i],v_re[i])/M_PI);
//            printf("|%s> (%5d) -> sqr ampl: %.3e\n",index.c_str(),i, norm(var));
       }
    }
    std::cout<<std::endl;
}


#ifdef SPARSE
void qoxo_print(double *v_re, double *v_im, std::vector<uint> actives){
#else
void qoxo_print(double *v_re, double *v_im, uint vecsize){
#endif


    struct winsize wsize;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &wsize);

    /* size.ws_row is the number of rows, size.ws_col is the number of columns. */


    // for non-contiguous even-odd entries corresponding to real and imag parts
    using  uintdoub = std::pair<uint,double>;
    std::vector<uintdoub> state_norm;
#ifdef SPARSE
    for(const uint& i : actives){
#else
    for(uint i=0; i<vecsize; ++i){
#endif
       std::complex<double> var(v_re[i],v_im[i]);
       if(norm(var)>1e-8){
           state_norm.push_back(std::make_pair(i,norm(var)));
       }
    }
    std::sort(state_norm.begin(),state_norm.end(),[](const uintdoub &a, const uintdoub &b){return a.second > b.second;});

    std::vector<std::vector<char>> gms(state_norm.size(),std::vector<char>(9));    
    for(size_t idx = 0; idx< state_norm.size(); ++idx){
        auto uid = state_norm[idx];
        uint game_stidx = uid.first;
        for(uint i=0U; i<9U; ++i){
            gms[idx][i] = qoxocharmap[(game_stidx & 3U)]; // 0 -> empty, 1-> first pl, 2-> second pl, 3-> inactive
            game_stidx>>=2U;
        }
    }

    printf("\n");
    size_t max_ngxrow = wsize.ws_col/12;
    size_t max_ngxcol = (state_norm.size()+max_ngxrow-1)/max_ngxrow;
    for(size_t Row = 0; Row<max_ngxcol;++Row){
        size_t ngxrow = (Row==max_ngxcol-1)?(state_norm.size()%max_ngxrow):max_ngxrow;

        for(size_t Col=0; Col<ngxrow; ++Col) printf("------------");
        printf("\n");

        for(size_t Col=0; Col<ngxrow; ++Col){
            size_t idx = Row*max_ngxrow+Col;
            printf("|   %c|%c|%c  |",gms[idx][0],gms[idx][1],gms[idx][2]);
        }
        printf("\n");
        for(size_t Col=0; Col<ngxrow; ++Col) printf("|   _____  |");
        printf("\n");

        for(size_t Col=0; Col<ngxrow; ++Col){
            size_t idx = Row*max_ngxrow+Col;
            printf("|   %c|%c|%c  |",gms[idx][3],gms[idx][4],gms[idx][5]);
        }
        printf("\n");
        for(size_t Col=0; Col<ngxrow; ++Col) printf("|   _____  |");
        printf("\n");

        for(size_t Col=0; Col<ngxrow; ++Col){
            size_t idx = Row*max_ngxrow+Col;
            printf("|   %c|%c|%c  |",gms[idx][6],gms[idx][7],gms[idx][8]);
        }
        printf("\n");
        for(size_t Col=0; Col<ngxrow; ++Col){
            size_t idx = Row*max_ngxrow+Col;
            printf("|  " ACYAN "%5.1f%%" ARESET "  |",state_norm[idx].second*100);
        }
        printf("\n");
        for(size_t Col=0; Col<ngxrow; ++Col) printf("------------");
        printf("\n");
        
    }
    std::cout<<std::endl;
}

void sparse_print(double *v_re, double *v_im, size_t size, std::vector<uint>& indexes){
    std::sort(indexes.begin(), indexes.end());
    // for non-contiguous even-odd entries corresponding to real and imag parts
    size_t index_size = (int)std::round(std::log2(size));
    for(const auto& idx : indexes){
       std::complex<double> var(v_re[idx],v_im[idx]);
		std::string index= std::bitset<32>(idx).to_string();
		index.erase(0,32-index_size);
		printf("|%s> (%5d) -> (%.3e, %.3e ) : mod2= %.3e, phase= %.3e pi\n",index.c_str(),idx, v_re[idx], v_im[idx], norm(var),atan2(v_im[idx],v_re[idx])/M_PI);
    }
    std::cout<<std::endl;
}

template<class T>
void print(std::vector<T> v){
    for(const auto& el : v)
        std::cout<<el<<" ";
    std::cout<<std::endl;
}

