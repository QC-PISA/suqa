import numpy as np
import sys

filestem=sys.argv[1]

a=np.random.random((8,8))+np.random.random((8,8))*1j
x = a+np.conjugate(a.T)


vals,vecs = np.linalg.eigh(x)

vecs=np.conjugate(vecs.T)

np.savetxt(filestem+"_matrix_re",np.real(x), delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_matrix_im",np.imag(x), delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_vals",vals, delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_vecs_re",np.real(vecs), delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(filestem+"_vecs_im",np.imag(vecs), delimiter=' ', newline='\n', header='', footer='', comments='# ')


print((np.abs(x - np.conjugate(vecs.T).dot(np.diag(vals).dot(vecs)))<1e-10).all())
