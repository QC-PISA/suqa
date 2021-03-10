#include "io.hpp"

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
            printf("|%s> (%5d) -> (%.3e, %.3e ) : mod2= %.3e\n",index.c_str(),i, (fabs(v_re[i])>1e-8)? v_re[i] : 0, (fabs(v_im[i])>1e-8)? v_im[i] : 0, norm(var));
//, phase= %.3e pi   ,atan2(v_im[i],v_re[i])/M_PI
       }
    }
    std::cout<<std::endl;
}

void sparse_print(double *v_re, double *v_im, size_t size, std::vector<uint>& indexes){
    std::sort(indexes.begin(), indexes.end());
    // for non-contiguous even-odd entries corresponding to real and imag parts
    size_t index_size = (int)std::round(std::log2(size));
    for(const auto& idx : indexes){
       std::complex<double> var(v_re[idx],v_im[idx]);
       if(norm(var)>1e-8){
		std::string index= std::bitset<32>(idx).to_string();
		index.erase(0,32-index_size);
		printf("|%s> (%5d) -> (%.3e, %.3e ) : mod2= %.3e\n",index.c_str(),idx, (fabs(v_re[idx])>1e-8)? v_re[idx] : 0, (fabs(v_im[idx])>1e-8)? v_im[idx] : 0,norm(var));
//, phase= %.3e pi   ,atan2(v_im[idx],v_re[idx])/M_PI
       }
    }
    std::cout<<std::endl;
}

template<class T>
void print(std::vector<T> v){
    for(const auto& el : v)
        std::cout<<el<<" ";
    std::cout<<std::endl;
}

