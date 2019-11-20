import numpy as np
import sys

if(len(sys.argv)<2):
    print("usage: python "+sys.argv[0]+" <filestem>")
    sys.exit(1)
filestem=sys.argv[1]
#
#a=np.random.random((size*2,size*2))+np.random.random((size*2,size*2))*1j
#x = a+np.conjugate(a.T)

Id=np.identity(2)
x=np.array([[0,1],[1,0]])
y=np.array([[0,-1j ],[1j,0]])
z=np.array([[1, 0], [0, -1]])
z2=1./np.sqrt(2)
h=z2*(x+z)

A=np.kron(x,np.kron(x,Id+y))

S_H=np.array([[z2,0,0,-z2,0,0,0,0],[0,z2,-z2,0,0,0,0,0],[0,0,0,0,z2,0,0,-z2],[0,0,0,0,0,z2,-z2,0],[.5,0,0,.5,0,-.5,-.5,0],[0,.5,.5,0,-.5,0,0,-.5],[.5,0,0,.5,0,.5,.5,0],[0,.5,.5,0,.5,0,0,.5]])

H=np.kron(x,np.kron(x,Id))+ np.kron(x,np.kron(Id,x)) + np.kron(Id,np.kron(x,x))
Ad=S_H.dot(A.dot(np.conjugate(S_H.T)))
if (H.dot(A)-A.dot(H)==0).all():
    print("ERROR!")

print("subtraces: ",np.trace(Ad[:6,:6]),np.trace(Ad[6:,6:]))
print("S_H.dot(A.dot(np.conjugate(S_H.T))) = ",S_H.dot(A.dot(np.conjugate(S_H.T))))

vals,vecs = np.linalg.eigh(A)

vecs=np.conjugate(vecs.T)

np.savetxt(filestem+"_matrix_re",np.real(A), delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_matrix_im",np.imag(A), delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_vals",vals, delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_vecs_re",np.real(vecs), delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_vecs_im",np.imag(vecs), delimiter=' ', newline='\n', header='', footer='', comments='# ')


print((np.abs(A - np.conjugate(vecs.T).dot(np.diag(vals).dot(vecs)))<1e-10).all())

Hs=np.array([0.,0.,0.,0.,0.,0.,1.,1.])
print("Hs:\n",Hs)
print("np.diag(np.exp(-1.0*Hs)):\n",np.diag(np.exp(-1.0*Hs)))
Xmat = A
def Z(b):
    return np.sum(np.exp(-np.outer(b, Hs)), axis=1)
bs=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
X_exact = 1./Z(bs) * np.array([np.real(np.trace(np.diag(np.exp(-b*Hs)).dot(S_H.dot(Xmat.dot(np.conjugate(S_H.T)))))) for b in bs])

print("X_exact = ",X_exact)
