import numpy as np
import sys

if(len(sys.argv)<3):
    print("usage: python "+sys.argv[0]+" <size> <filestem>")
    sys.exit(1)
size = int(sys.argv[1])
filestem=sys.argv[2]

a=np.random.random((size*2,size*2))+np.random.random((size*2,size*2))*1j
x = a+np.conjugate(a.T)


vals,vecs = np.linalg.eigh(x)

vecs=np.conjugate(vecs.T)

np.savetxt(filestem+"_matrix_re",np.real(x), delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_matrix_im",np.imag(x), delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_vals",vals, delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_vecs_re",np.real(vecs), delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_vecs_im",np.imag(vecs), delimiter=' ', newline='\n', header='', footer='', comments='# ')


print((np.abs(x - np.conjugate(vecs.T).dot(np.diag(vals).dot(vecs)))<1e-10).all())
