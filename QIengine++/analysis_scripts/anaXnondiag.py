import matplotlib.pyplot as plt
import sys
import numpy as np

if len(sys.argv)<2:
    print("needs file stem")
    sys.exit(1)

fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_xlabel(r"$\beta$")
ax2.set_xlabel(r"$\beta$")
ax1.set_xlabel(r"Averages")
ax2.set_xlabel(r"Discrepancy")

bs=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
aa=[np.loadtxt(sys.argv[1]+"_b_"+str(vl)) for vl in bs]
vs=[np.mean(ael[:,1]) for ael in aa]
ss=[np.std(ael[:,1])/np.sqrt(len(ael)) for ael in aa]

Hs=np.array([1.,0.,2.])
def Z(b):
    return np.sum(np.exp(-np.outer(b, Hs)),axis=1)
ene_exact=np.sum(np.exp(-np.outer(bs, Hs))*Hs,axis=1)/Z(bs)
ax1.plot(bs,ene_exact, label=r'$\langle E \rangle(\beta)$ (exact)')
ax1.errorbar(bs,vs,ss,linestyle="",capsize=3, label=r'$\langle E \rangle(\beta)$  (data)')

ax2.plot(bs,0.0*bs, label=r'Baseline')
ax2.errorbar(bs,vs-ene_exact, ss,linestyle="",capsize=3, label=r'$\langle E \rangle_{data} - \langle E \rangle_{exact}$')

if aa[0].shape[1]>2:
    vsX=[np.mean(ael[:,2]) for ael in aa]
    ssX=[np.std(ael[:,2])/np.sqrt(len(ael)) for ael in aa]
    X_exact = np.exp(-bs)/Z(bs)
    ax1.plot(bs,X_exact, label=r'$\langle X \rangle(\beta)$ (exact)')
#    plot(bs,(np.exp(-bs)+5./2.*np.exp(-bs*0.0)+5./2.*np.exp(-2*bs))/Z(bs), label=r'X operator')
    ax1.errorbar(bs,vsX,ssX,linestyle="",capsize=3, label=r'$\langle X \rangle(\beta)$ (data)')
    ax2.errorbar(bs,vsX-X_exact, ssX,linestyle="",capsize=3, label=r'$\langle X \rangle_{data} - \langle X \rangle_{exact}$')
ax1.legend()
ax2.legend()
plt.tight_layout()
fig.savefig(sys.argv[1]+"_analysis.pdf",dpi=200)
plt.show()
