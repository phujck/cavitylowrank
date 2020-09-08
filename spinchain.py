##########################################################
# Simulation to compare low-rank approximation to exact  #
# and Monte-Carlo solutions to Lindblad equation for a   #
# Heisenberg spin-chain. Code is based off the QuTiP     #
# example here: https://bit.ly/342ZwC5                   #
# USE THIS FILE ONLY TO ACTUALLY RUN SIMULATION.         #
# SET SYSTEM PARAMETERS IN param.py                      #
# VIEW RESULTS IN analysis.py
##########################################################

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
import time
# Grab parameters from params file.
from params import *
import os

cite()
params = {
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [2 * 3.375, 2 * 3.375],
    'text.usetex': True,
    'figure.figsize': (16, 12)
}

plt.rcParams.update(params)
# Set some resource limits. Unfortunately the monte-carlo simulations don't seem to give a shit about the thread
# assignment here, and uses every core it can get its hands on. To fix that I'd have to go digging into the
# multiprocessing library files, which doesn't seem best pracrice. Just worth bearing in mind that MC is getting
# access to a lot more resources!
threads = 20
os.environ['NUMEXPR_MAX_THREADS'] = '{}'.format(threads)
os.environ['NUMEXPR_NUM_THREADS'] = '{}'.format(threads)
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)

"""Function for setting up Heisenberg Spin Chain"""


def integrate_setup(N, fock_N, kappa, g, beta, gamma, deltas):
    si = qeye(2)
    ai=qeye(fock_N)
    sz = sigmaz()
    sp = sigmap()
    sm = sigmam()

    op_list=[]
    for m in range(N):
        op_list.append(si)
    si_tensor=tensor(op_list)
    a=tensor(si_tensor,destroy(fock_N))
    number_a=a.dag()*a
    sm_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)
        op_list.append(ai)

        op_list[n] = sm
        sm_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))


    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H+=g*(a.dag()*sm_list[n]+sm_list[n].dag()*a)
        for m in range(N):
            H += deltas[n,m] * sm_list[n].dag()*sm_list[m]
    H+=np.sqrt(kappa)*beta*(a+a.dag())



    # collapse operators
    c_op_list = []

    # spin dephasing
    if kappa > 0.0:
        c_op_list.append(np.sqrt(kappa)*a)

    # adding coupling to bath:
    if gamma > 0:
        sm_sum = 0
        for n in range(N):
            sm_sum += sm_list[n]
        c_op_list.append(np.sqrt(gamma)*sm_sum)

    print('Hamiltonian and dissipators constructed. shape of H is {}, and Hermitian check returned {}'.format(H.shape,H.isherm))

    return H, sz_list, [number_a] , c_op_list


"""Helper function for switching between exact and monte-carlo qutip solvers"""


def qutip_solver(H, psi0, tlist, c_op_list, sz_list, solver, ntraj=1000):
    # evolve and calculate expectation values
    if solver == "me":
        result = mesolve(H, psi0, tlist, c_op_list, sz_list, progress_bar=True)
    elif solver == "mc":
        result = mcsolve(H, psi0, tlist, c_op_list, sz_list, ntraj, progress_bar=True)

    return result.expect


"""Sets up unitaries required for low-rank approximation"""


def unitaries(H, c_op_list, dt):
    K = len(c_op_list)
    print(K)
    U_ops = []
    if K == 0:
        U_ops.append((-1j * H * dt).expm())
    else:
        prefactor = np.sqrt(1 / (2 * K))
        for op in c_op_list:
            exponent = -1j * H * dt
            if op.isherm == False:
                exponent += (dt * K / 2) * (op * op - op.dag() * op)

            # exponent += (dt * K / 2) * (op * op - op.dag() * op)
            extra = 1j * np.sqrt(K * dt) * op
            P1 = exponent + extra
            P2 = exponent - extra
            U_prop = prefactor * (P1.expm())
            V_prop = prefactor * (P2.expm())
            U_ops.append(U_prop)
            U_ops.append(V_prop)
    return U_ops


"""trying a version of unitaries that incorporates richardson extrapolation."""
# def unitaries(H, c_op_list, dt):
#     K = len(c_op_list)
#     print(K)
#     U_ops = []
#     step_factor = 2
#     error_order = 1
#     dt2=dt/step_factor
#     divider=1/((step_factor**error_order)-1)
#     if K == 0:
#         U_ops.append((-1j * H * dt).expm())
#     else:
#         prefactor = np.sqrt(1 / (2 * K))
#         for op in c_op_list:
#             exponent = -1j * H * dt
#             exponent_t=-1j * H * dt2
#             if op.isherm == False:
#                 exponent += (dt * K / 2) * (op * op - op.dag() * op)
#                 exponent_t += (dt2 * K / 2) * (op * op - op.dag() * op)
#
#             # exponent += (dt * K / 2) * (op * op - op.dag() * op)
#             extra = 1j * np.sqrt(K * dt) * op
#             extra_t = 1j * np.sqrt(K * dt2) * op
#
#             P1 = exponent + extra
#             P2 = exponent - extra
#             P1_t = exponent_t + extra_t
#             P2_t = exponent_t - extra_t
#             U_prop = prefactor * (divider*((step_factor**error_order)*P1_t.expm()-P1.expm()))
#             V_prop = prefactor * (divider*((step_factor**error_order)*P2_t.expm()-P2.expm()))
#             U_ops.append(U_prop)
#             U_ops.append(V_prop)
#     return U_ops




"""Calculates overlap matrix for truncating the set of wavefunctions in low-rank approx."""
"""OLD VERSION. SLOWER!"""


def overlap(psi_list):
    matrix = np.zeros((len(psi_list), len(psi_list)), dtype=np.complex_)
    for i in range(len(psi_list)):
        for j in range(i, len(psi_list)):
            # overlap=psi_list[i].dag()*psi_list[j]
            # print('method 1')
            # print(overlap.data)
            # print('method 2')
            o2 = psi_list[i].overlap(psi_list[j])
            # print(o2)
            matrix[i, j] = o2
            if j > i:
                matrix[j, i] = np.conj(o2)
    # print(matrix)
    return matrix


"""alternate way of calculating overlap, isn't nearly as wasteful in terms of indexing, and can be decorated with njit"""


@njit
def alt_overlap(psi_list):
    matrix = psi_list.conj() @ psi_list.T
    return matrix  # psi_list.append(basis(2,1))


# for n in range(N-1):
#     psi_list.append(basis(2,0))


"""The orthogonalisation procedure the overlaps are used in"""


def orthogonalise(psi_list, U, zeroobj, rank):
    g_list = []
    U = U.T
    if rank > len(psi_list):
        rank = len(psi_list)
    norm = 0
    for i in range(rank):
        g = zeroobj
        for j in range(len(psi_list)):
            # print(U[i,j])
            # print(psi_list[j].dims)
            g_unit = U[i, j] * psi_list[j]
            g += g_unit
        # norm+=g.norm()**2
        g_list.append(g)
    # g_list=[g/np.sqrt(norm) for g in g_list]
    return g_list


"""Performs a single step in the low-rank approximation. Returns a list of arrays which are the set of psis evolved by one step"""


def one_step(psis, U_ops, zeropsi, rank):
    new_psis = []
    for psi in psis:
        for U in U_ops:
            newpsi = U * psi
            new_psis.append(newpsi)
    if len(new_psis) > rank:
        # mat = overlap(new_psis)
        # This is a bit awkward, needing to convert from a list of arrays to a 2d np array in this way, but it's the
        # first working solution I found.
        psi_array = np.array([psi.full() for psi in new_psis])
        # print(psi_array.shape)
        mat = alt_overlap(psi_array.squeeze())
        w, v = np.linalg.eigh(mat)
        v = v[:, ::-1]
        new_psis = orthogonalise(new_psis, v, zeropsi, rank)
    return new_psis


"""Calculates expectations from the set of low-rank psis."""


def expectations(psis, ops, normalise):
    expecs = []
    for op in ops:
        expec = 0
        norm = 0
        for psi in psis:
            expec += expect(op, psi)
            if normalise:
                norm += psi.norm()**2
        if normalise:
            expec = expec / np.sqrt(norm)
        expecs.append(expec)
    return expecs


"""Runs the exact Lindblad equation, saves the results."""


def exact_sim(run_exact):
    outfile_exact = './Data/exact:{}sites-{}cavityN-init{}-{}g-{}beta-{}kappa-{}gamma-{}detune-{}shift.npz'.format(
        N, fock_N,init,g,beta,kappa,gamma,detune,shift)
    if run_exact:
        loop_start = time.time()
        sz_expt = qutip_solver(H, psi0, tlist, c_op_list, sz_list + sx_list, "me")
        loop_end = time.time()
        expect_dict = dict()
        expect_dict['expectations'] = sz_expt
        expect_dict['runtime'] = loop_end - loop_start
        print('loop time {}'.format(expect_dict['runtime']))
        np.savez(outfile_exact, **expect_dict)


"""Runs Monte-Carlo Lindblad, saves the results."""


def mc_sim(ntraj, run_mc):
    outfile_mc = './Data/montecarlo:{}sites-{}cavityN-init{}-{}g-{}beta-{}kappa-{}gamma-{}detune-{}shift-{}traj.npz'.format(
        N, fock_N,init,g,beta,kappa,gamma,detune,shift,ntraj)

    if run_mc:
        loop_start = time.time()
        sz_expt = qutip_solver(H, psi0, tlist, c_op_list, sz_list + sx_list, "mc", ntraj)
        loop_end = time.time()
        expect_dict = dict()
        expect_dict['expectations'] = sz_expt
        expect_dict['runtime'] = loop_end - loop_start
        print('loop time {}'.format(expect_dict['runtime']))
        np.savez(outfile_mc, **expect_dict)


"""Runs low-rank simulation, saves the results."""


def lowrank_sim(rank, run_lowrank):
    outfile_lowrank = './Data/lowrank:{}sites-{}cavityN-init{}-{}g-{}beta-{}kappa-{}gamma-{}detune-{}shift-{}rank.npz'.format(
        N, fock_N,init,g,beta,kappa,gamma,detune,shift,rank)
    if run_lowrank:
        loop_start = time.time()
        psis = [psi0]
        s_rank = []
        psi_times = []
        time_psis = []
        for j in tqdm(range(steps)):
            # s_rank.append(expectations(psis,sz_list+sx_list))
            time_psis.append(psis)
            psis = one_step(psis, U_ops, zeropsi, rank)
            # if j%100==0:
            #     norms=[psi.norm()**2 for psi in psis]
            #     norm=np.sum(norms)
            #     psis = [psi/np.sqrt(norm) for psi in psis]
            # norms=[psi.norm()**2 for psi in psis]
            # norm=np.sum(norms)
            # psis = [psi/np.sqrt(norm) for psi in psis]


        loop_end = time.time()
        # psi_list = [psi for psi in psis]
        print('Low rank propagation done. Time taken %.5f seconds' % (loop_end - loop_start))

        for psis in time_psis:
            s_rank.append(expectations(psis, sz_list + sx_list, False))
        s_rank = np.array(s_rank)
        expect_dict = dict()
        expect_dict['expectations'] = s_rank
        expect_dict['runtime'] = loop_end - loop_start
        print('loop time {}'.format(expect_dict['runtime']))
        np.savez(outfile_lowrank, **expect_dict)


"""This takes the parameters from the params file"""
# driving
# intial state, first spin in state |1>, the rest in state |0>
psi_list = []
zero_list = []
zero_ket = Qobj([[0], [0]])
fock_zero_list=[]
for n in range(fock_N):
    fock_zero_list.append([0])
fock_zero_ket=Qobj(fock_zero_list)


start = time.time()

for n in range(N):
    # if n%2==0:
    #     psi_list.append(basis(2,0))
    # else:
    #     psi_list.append(basis(2,1))
    psi_list.append((basis(2,1)+basis(2,0))/np.sqrt(2))
    # psi_list.append(basis(2, 1))
    zero_list.append(zero_ket)

psi_list.append(basis(fock_N,0))
# psi_list.append(basis(fock_N,2))

zero_list.append(fock_zero_ket)
psi0 = tensor(psi_list)
zeropsi = tensor(zero_list)
# print(zeropsi)
print(deltas)
H, sz_list, sx_list, c_op_list = integrate_setup(N, fock_N, kappa, g, beta,gamma, deltas)
U_ops = unitaries(H, c_op_list, deltat)
end = time.time()
print('operators built! Time taken %.3f seconds' % (end - start))
psis = [psi0]

"""Single run for simulations"""
# exact_sim(run_exact)
# mc_sim(ntraj,run_mc)
# lowrank_sim(rank,run_lowrank)


# """batch runs"""
for fock_N in [10]:
    for N in [7]:
        # counter=0
        for kappa_var in [1e-1]:
            for gamma_var in [1e-3]:
                for shift_var in [0.1]:
                    kappa = kappa_var * g
                    gamma = gamma_var * g
                    beta=kappa
                    # shift=1e-1*g
                    shift = shift_var * detune
                    delta_column=[]
                    for n in range(N):
                        delta_row = []
                        for m in range(N):
                            if m == n:
                                delta_row.append(detune * (m + 1))
                            else:
                                delta_row.append(shift)
                        delta_column.append(delta_row)
                    deltas = np.array(delta_column)
                    print('now running for {}-sites, {:.2e}-kappa, {:.2e}-Gamma,{:.2e}-Lambd shift'.format(N, kappa_var, gamma_var,shift_var))
                    psi_list = []
                    zero_list = []
                    zero_ket = Qobj([[0], [0]])
                    fock_zero_list = []
                    for n in range(fock_N):
                        fock_zero_list.append([0])
                    fock_zero_ket = Qobj(fock_zero_list)

                    start = time.time()

                    for n in range(N):
                        psi_list.append((basis(2,1)+basis(2,0))/np.sqrt(2))
                        # psi_list.append(basis(2, 0))
                        zero_list.append(zero_ket)
                    psi_list.append(basis(fock_N, 0))
                    zero_list.append(fock_zero_ket)
                    psi0 = tensor(psi_list)
                    zeropsi = tensor(zero_list)
                    # print(zeropsi)
                    print(deltas)
                    H, sz_list, sx_list, c_op_list = integrate_setup(N, fock_N, kappa, g, beta, gamma, deltas)
                    U_ops = unitaries(H, c_op_list, deltat)
                    end = time.time()
                    print('operators built! Time taken %.3f seconds' % (end - start))
                    psis = [psi0]
                    # if fock_N==10:
                    exact_sim(run_exact)
                    for ntraj in [10000]:
                        print('now running for {}-sites, {:.2e}-kappa, {:.2e}-Gamma,{:.2e}-Lambd shift'.format(N,
                                                                                                               kappa_var,
                                                                                                               gamma_var,
                                                                                                               shift_var))
                        print('now running monte-carlo {}-trajectories'.format(ntraj))
                        mc_sim(ntraj, run_mc)
                    for rank in [1,2,4,8]:
                        print('now running for {}-sites, {:.2e}-kappa, {:.2e}-Gamma,{:.2e}-Lambd shift'.format(N, kappa_var,
                                                                                                               gamma_var,
                                                                                                               shift_var))
                        print('now running low-rank at rank {}'.format(rank))
                        lowrank_sim(rank, run_lowrank)



"""Old Plotting Code. The New Code saves these expectations for plotting in analysis.py"""
# print(s_rank.size)
# # print(np.array(s_rank))
# print(s_rank[:,0])
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
# sz_lr = np.zeros(len(sz_expt[0]))
# sz_e = np.zeros(len(sz_expt[0]))
# sx_lr = np.zeros(len(sz_expt[0]))
# sx_e = np.zeros(len(sz_expt[0]))
# eps_z_lr=np.zeros(len(sz_expt[0]))
# eps_z_mc=np.zeros(len(sz_expt[0]))
# for n in range(N):
#     eps_z_lr+= np.sqrt((sz_expt[n]-s_rank[:,n])**2)
#     eps_z_mc+= np.sqrt((sz_expt[n]-sz_expt_mc[n])**2)
#     sz_lr += sz_expt[n]
#     sx_lr += sz_expt[N+n]
#     sx_e+=s_rank[:,N + n]
#     sz_e+=s_rank[:,n]

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
# for n in range(N):
#     ax1.plot(tlist, np.real(sz_expt[n]), label='$\\langle\\sigma_z^{(%d)}\\rangle$' % n)
#     ax1.plot(tlist, np.real(s_rank[:,n]), linestyle='--',label='Low-rank $\\langle\\sigma_z^{(%d)}\\rangle$' % n)
#     ax2.plot(tlist, np.real(sz_expt[N + n]), label='$\\langle\\sigma_x^{(%d)}\\rangle$' % n)
#     ax2.plot(tlist, np.real(s_rank[:,N + n]),linestyle='--', label='Low-rank $\\langle\\sigma_x^{(%d)}\\rangle$' % n)
# # ax1.legend(loc=0)
# ax2.set_xlabel('Time [ns]')
# ax1.set_ylabel('$\\langle\sigma_z\\rangle$')
# ax2.set_ylabel('$\\langle\sigma_x\\rangle$')
# ax2.legend()

# plt.show()
#
# # plt.title('simulation error')
# plt.plot(tlist,eps_z_lr, label='low-rank')
# plt.plot(tlist,eps_z_mc, label='monte-carlo')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('$\\epsilon(t)$')
# plt.show()
#
# plt.subplot(211)
# plt.plot(tlist, sz_e,label='low-rank approx')
# plt.plot(tlist, sz_lr, linestyle='dashed',color='black',label='exact')
# plt.legend()
# plt.ylabel('$\\langle\sigma_z\\rangle$')
#
# plt.subplot(212)
# plt.plot(tlist, sx_lr,label='low-rank approx')
# plt.plot(tlist, sx_e, linestyle='dashed',color='black',label='exact')
# plt.legend()
# plt.ylabel('$\\langle\sigma_x\\rangle$')
# plt.xlabel('Time [ns]')
# plt.show()
#
# final_sz = [sz_expt[n][-1] for n in range(N)]
# plt.plot(range(N), final_sz)
# plt.show()
