/**
 * \file Rand.hpp
 * Lattice class definition.
 *
 * \brief Implements wrapper classes for random number generators.
 *
 * \author Giuseppe Clemente <giuseppe.clemente93@gmail.com> 
 * \version 1.0
 */
#pragma once
#include <fstream>
#include <string>
#include <math.h>
#include <sys/time.h>
#include <iostream>
#include "pcg32.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
constexpr auto M_PI = 3.141592653589793238462643383279502884L;
#endif

typedef unsigned long long int Ullong;
typedef unsigned int Uint;
typedef long double Ldoub;

struct Ran_state{
	Ullong u,v,w;
};

/** 
 * \brief Ran generator from Numerical Recipes.
 *
 * Implementation of the highest quality recommended generator. The constructor is called with an integer seed and creates an instance of the generator. The member functions int64, doub, and int32 return the next values in the random sequence, as a variable type indicated by their names. The period of the generator is ~3.138x10^57.
 */
class Ran {
		Ullong u,v,w;
		bool bGauss_cached=false;
		Ldoub gauss_cache;
	public:
        Ran() : Ran(0ULL) {}
		Ran(Ullong j) : v(4101842887655102017ULL), w(1) {
			if(j==0){
				struct timeval tval;
				gettimeofday(&tval,NULL);
				j=tval.tv_sec*1000000ULL+tval.tv_usec;
			}
			u = j ^ v; int64();
			v = u; int64();
			w = v; int64();
		}
		Ran(Ran_state ran_st){
			u=ran_st.u;
			v=ran_st.v;
			w=ran_st.w;
		}
		
		~Ran() {}
		
		Uint int32() { 
			return (Uint)int64(); 
		}
		
		Ran_state get_state(){
			Ran_state ret;
			ret.u=u;
			ret.v=v;
			ret.w=w;
			return ret;
		}
		
		void restart(){
			struct timeval tval;
			gettimeofday(&tval,NULL);
			Ullong j=tval.tv_sec*1000000ULL+tval.tv_usec;
			v=4101842887655102017ULL;
			w=1;
			u = j ^ v; int64();
			v = u; int64();
			w = v; int64();
		}
				

		
		Ullong int64(){
			u = u* 286293355577941757ULL + 7046029254386353087ULL;
			v ^= v >> 17; v^= v << 31; v ^= v >> 8;
			w = 4294957665U*(w & 0xffffffff) + (w >> 32);
			Ullong x = u ^ (u <<21); x ^= x >>35; x^= x<<4;
			return (x + v) ^ w;
		}
		
		Ldoub doub(){ 
			return 5.42101086242752217E-20 * int64(); 
		}
		
/* 		<template T> T two_point(T x1, T x2, double p1){
			return (doub()<p1) ? x1 : x2;
		} */
		
		double two_point(double x1, double x2, double p1){
			return (doub()<p1) ? x1 : x2;
		}
		
		int randint(int x0, int x1){ //estremo destro non compreso
			return x0+int64()%(x1-x0);
		} 
		
		double exponential(double tau){
			return -log(doub())*tau;
		}		
		
		Ldoub gauss_BoxMuller(double x0, double sig){
			return x0+sig*gauss_standard_BoxMuller();
		}
		
		Ldoub gauss_PolarMarsaglia(double x0, double sig){
			return x0+sig*gauss_standard_PolarMarsaglia();
		}
		
		Ldoub gauss_standard_BoxMuller(){
			if (bGauss_cached){
				bGauss_cached=false;
				return gauss_cache;
			}
			bGauss_cached=true;
			Ldoub v1=sqrtl(-2*log(doub()));
			Ldoub v2=2*M_PI*doub();
			gauss_cache=v1*cos(v2);
			return v1*sin(v2);
		}
		
		Ldoub gauss_standard_PolarMarsaglia(){
			if (bGauss_cached){
				bGauss_cached=false;
				return gauss_cache;
			}
			bGauss_cached=true;
			Ldoub v1=2*doub()-1;
			Ldoub v2=2*doub()-1;
			Ldoub w=powl(v1,2)+powl(v2,2);
			bool bFlipFirst=true;
			while(w>1){
				if(bFlipFirst){
					v1=2*doub()-1;
					bFlipFirst=false;
				}else{
					v2=2*doub()-1;
					bFlipFirst=true;
				}
				w=powl(v1,2)+powl(v2,2);
			}
			Ldoub tmp_factor=sqrtl(-2*log(w)/w);
			gauss_cache=v1*tmp_factor;
			return v2*tmp_factor;
		}	
};
/** 
 * \brief Ranq1 generator from Numerical Recipes.
 *
 * Recommended generator for everyday use. The period is ~8x10^19. Calling conventions same as Ran, above.
 */
class Ranq1 {
		Ullong v;
	public:
		Ranq1(Ullong j) : v(4101842887655102017ULL) {
				v ^= j;
				v = int64();
		}
		
		~Ranq1() {}
		
		int randint(int x0, int x1){
			return x0+int64()%(x1-x0);
		}
		
		Uint int32(){ 
			return  (Uint)int64(); 
		}
		
		Ullong int64(){
				v ^= v >> 21; v ^= v << 35; v^= v >> 4;
				return v * 2685821657736338717ULL;
		}
		
		Ldoub doub(){ 
			return 5.42101086242752217E-20 * int64(); 
		}
};

/** 
 * \brief Ranq2 generator from Numerical Recipes.
 *
 * Backup generator if Ranq1 has too short a period and Ran is too slow. The period is ~8.5x10^37. Calling conventions same as Ran, above.
 */

class Ranq2 {
		Ullong v,w;
	public:
		Ranq2(Ullong j) : v(4101842887655102017ULL), w(1) {
			v ^= j;
			w = int64();
			v = int64();
		}
	
		~Ranq2() {}
		
		int randint(int x0, int x1){
			return x0+int64()%(x1-x0);
		}

		Uint int32(){ 
			return  (Uint)int64(); 
		}
	
		Ullong int64(){
			v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
			w = 4294957665U*(w & 0xffffffff) + (w >> 32);
			return v ^ w;
		}
		
		Ldoub doub(){ 
			return 5.42101086242752217E-20 * int64(); 
		}
};



/** 
 * \brief state struct for the pcg32 generator.
 */
struct pcg_state{
	uint64_t state;
};
/** 
 * \brief pcg32 generator of the pcg type.
 *
 * Very recommended generator. Period: 2^64, 32-bit precision.  
 */
class pcg{
    pcg32 rng;

    public:
        pcg() : pcg(0ULL) {}

        pcg(Ullong j){
            if(j==0){
			struct timeval tval;
			gettimeofday(&tval,NULL);
			j=tval.tv_sec*1000000ULL+tval.tv_usec;
            }
            rng.seed(j);
        }

        pcg(pcg_state pcg_st){
           rng.state = pcg_st.state; 
        }

        double doub(){
            return rng.nextDouble();
        }

		int randint(int x0, int x1){ //estremo destro non compreso
			return x0+rng.nextUInt()%(x1-x0);
		} 
		
		pcg_state get_state(){
			pcg_state ret;
			ret.state=rng.state;
			return ret;
		}
};






//namespace xoroshiro{
//#ifdef __GNUG__
//#pragma GCC diagnostic ignored "-Wsign-compare"
//#endif
//extern "C"
//{
//#include "xoroshiro128plus.c"
//
//}
//#ifdef ___GNUG__
//#pragma GCC diagnostic pop
//#endif
//}
//class xoroshiro128plus {
//    // See the description in file "xoroshiro128plus.c"
//	public:
//		xoroshiro128plus(Ullong j) : s{UINT64_C(10532447193056740057), UINT64_C(15725061932195978535)} {}
//};
