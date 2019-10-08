from pylab import *
import sys
if len(sys.argv)<2:
    print("needs file stem")
    sys.exit(1)
bs=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
aa=[np.loadtxt(sys.argv[1]+"_b_"+str(vl)) for vl in bs];
vs=[np.mean(ael[:,1]) for ael in aa]
ss=[np.std(ael[:,1])/np.sqrt(len(ael)) for ael in aa]

Hs=np.array([1.,0.,2.])
def Z(b):
    return np.sum(exp(-np.outer(b, Hs)),axis=1)
plot(bs,np.sum(exp(-np.outer(bs, Hs))*Hs,axis=1)/Z(bs), label=r'Energy')
errorbar(bs,vs,ss,linestyle="",capsize=3)

if aa[0].shape[1]>2:
    vsX=[np.mean(ael[:,2]) for ael in aa]
    ssX=[np.std(ael[:,2])/np.sqrt(len(ael)) for ael in aa]
#    plot(bs,exp(-bs)/Z(bs), label=r'X operator')
    plot(bs,(exp(-bs)+5./2.*exp(-bs*0.0)+5./2.*exp(-2*bs))/Z(bs), label=r'X operator')
    errorbar(bs,vsX,ssX,linestyle="",capsize=3)
legend()
show()
