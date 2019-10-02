import numpy as np
# import matplotlib
# from math import pi
import sys
import os

from qiskit import BasicAer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute

beta = 1.0
eps = 1.0
th1 = 2*np.arcsin(np.exp(-beta*eps))
th2 = 2*np.arcsin(np.exp(-2*beta*eps))
au_Phi = (1.+np.sqrt(5))/2.
au_phi = -1./au_Phi
au_sqPhi = np.sqrt(2+au_Phi)
au_sqphi = np.sqrt(2-au_Phi)


def toffolone_4(qc, qT, q0, q1, q2, q3, qaux_0, qaux_1, qaux_2):
    qc.barrier()
    qc.ccx(q0, q1, qaux_0)
    qc.ccx(q2, qaux_0, qaux_1)
    qc.ccx(q3, qaux_1, qaux_2)
    qc.cx(qaux_2, qT)
    qc.ccx(q3, qaux_1, qaux_2)
    qc.ccx(q2, qaux_0, qaux_1)
    qc.ccx(q0, q1, qaux_0)
    qc.barrier()


def draw_C():
    if np.random.rand() < 0.5:
        return 0
    else:
        return 1


def apply_C(qc, psi, Ci):
    if Ci == 0:
        qc.x(psi[0])
    elif Ci == 1:
        qc.x(psi[0])
    else:
        raise "ERROR"


def apply_C_inverse(qc, psi, Ci):
    apply_C(qc, psi, Ci)


def apply_Phi(qc, psi, ene_new):
    qc.cx(psi[0], ene_new[0])
    qc.cx(psi[1], ene_new[1])


def apply_Phi_inverse(qc, psi, ene_new):
    apply_Phi(qc, psi, ene_new)


def apply_W(qc, acc, ene_old, ene_new, qaux):
    # TODO: optimize gates

    # Eold  Enew
    # 0     1
    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[1])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_new[1])
    qc.x(ene_old[1])
    qc.x(ene_old[0])

    qc.cry(th1, qaux[3], acc[0])

    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[1])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_new[1])
    qc.x(ene_old[0])
    qc.x(ene_old[1])

    # 1     2
    qc.x(ene_old[1])
    qc.x(ene_new[0])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_old[1])
    qc.x(ene_new[0])

    qc.cry(th1, qaux[3], acc[0])

    qc.x(ene_new[0])
    qc.x(ene_old[1])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_new[0])
    qc.x(ene_old[1])

    # 0     2
    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[0])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[0])

    qc.cry(th2, qaux[3], acc[0])

    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[0])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[0])


    qc.x(acc[0])

    # 0     2
    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[0])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[0])

    qc.cry(-th2, qaux[3], acc[0])

    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[0])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[0])

    # 1     2
    qc.x(ene_old[1])
    qc.x(ene_new[0])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_old[1])
    qc.x(ene_new[0])

    qc.cry(-th1, qaux[3], acc[0])

    qc.x(ene_new[0])
    qc.x(ene_old[1])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_new[0])
    qc.x(ene_old[1])

    # 0     1
    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[1])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_new[1])
    qc.x(ene_old[1])
    qc.x(ene_old[0])

    qc.cry(-th1, qaux[3], acc[0])

    qc.x(ene_old[0])
    qc.x(ene_old[1])
    qc.x(ene_new[1])
    toffolone_4(qc, qaux[3], ene_old[0], ene_old[1], ene_new[0], ene_new[1], qaux[0], qaux[1], qaux[2])
    qc.x(ene_new[1])
    qc.x(ene_old[0])
    qc.x(ene_old[1])


def apply_W_inverse(qc, acc, ene_old, ene_new, qaux):
    apply_W(qc, acc, ene_old, ene_new, qaux)


def apply_U(qc, psi, ene_new, ene_old, acc, qaux, C_idx):
    apply_C(qc, psi, C_idx)
    apply_Phi(qc, psi, ene_new)
    apply_W(qc, acc, ene_old, ene_new, qaux)


def apply_U_inverse(qc, psi, ene_new, ene_old, acc, qaux, C_idx):
    apply_W_inverse(qc, acc, ene_old, ene_new, qaux)
    apply_Phi_inverse(qc, psi, ene_new)
    apply_C_inverse(qc, psi, C_idx)


psi = QuantumRegister(2, 'psi')
ene_old = QuantumRegister(2, 'ene_old')
ene_new = QuantumRegister(2, 'ene_new')
acc = QuantumRegister(1, 'acc')

qaux = QuantumRegister(4, 'qaux')

c_acc = ClassicalRegister(1, 'c_acc')
c_ene_old = ClassicalRegister(2, 'c_ene_old')
c_ene_new = ClassicalRegister(2, 'c_ene_new')


qc = QuantumCircuit(psi, ene_old, ene_new, acc, qaux, c_acc, c_ene_old, c_ene_new)

# initialize state (already done here)

# phase estimation (simply read state here)

# build U=W Phi C matrix, with C drawn randomly



def initialization():
    qc.measure(ene_old, c_ene_old)


def iteration():
    qc.reset(acc)
    qc.reset(ene_new)
    qc.reset(ene_old)
    apply_Phi(qc, psi, ene_old)

    C_idx = draw_C()
    apply_U(qc, psi, ene_new, ene_old, acc, qaux, C_idx)

    qc.measure(acc, c_acc)

    if c_acc == 1:
        print("accepted")
        return

    # else

    apply_U_inverse(qc, psi, ene_new, ene_old, acc, qaux, C_idx, c_acc)

    iters = 50
    while iters > 0:
        apply_Phi(qc, psi, ene_new, c_acc)
        qc.measure(ene_old, c_ene_old)
        qc.measure(ene_new, c_ene_new)
        apply_Phi_inverse(qc, psi, ene_new)

        if np.array_equal(c_ene_old, c_ene_new):
            # P1
            print("accepted")
            break
        else:
            # Q_alpha
            apply_U(qc, psi, ene_new, ene_old, acc, qaux, C_idx)
            qc.measure(acc, c_acc)
            apply_U_inverse(qc, psi, ene_new, ene_old, acc, qaux, C_idx)

            iters -= 1

    if iters == 0:
        print("not converged :(")
        sys.exit(1)

# def measure_observable():


num_sim_iters = 100

num_iters = 100

backend=BasicAer.get_backend('qasm_simulator')
energies = []
if __name__ == '__main__':
    for j in range(0, num_sim_iters):
        qc = QuantumCircuit(psi, ene_old, ene_new, acc, qaux, c_acc, c_ene_old, c_ene_new)
        initialization()
        for i in range(0, num_iters):
            iteration()
        
        qc.reset(ene_old)
        apply_Phi(qc, psi, ene_old)
        qc.measure(ene_old, c_ene_old)
        job=execute(qc,backend) # .result().get_unitary(qc,decimals=2)[:,i] 
        energies.append(c_ene_old)
        
np.savetxt("energies.txt",np.array(energies))
#    measure_observable()


