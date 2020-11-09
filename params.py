import numpy as np
fock_N=10
N = 7 # number of spins
# array of spin energy splittings and coupling strengths. here we use
# uniform parameters, but in general we don't have to

# init='mixed'
init='xbasis'
# dephasing rate
# gamma = 0.01 * np.ones(N)


g=0.05
kappa=1e-1*g
beta=kappa
# beta=0
gamma=1e-3*g
detune= 1
# shift=2*gamma
shift=0
delta_column=[]
for n in range(N):
    delta_row=[]
    for m in range(N):
        if m ==n:
            delta_row.append(detune*(m+1))
            # delta_row.append(detune)
        else:
            delta_row.append(shift)
    delta_column.append(delta_row)
deltas=np.array(delta_column)

steps =2000
steps2=2000
endtime=100
endtime2=100

tlist, deltat = np.linspace(0, endtime, steps, retstep=True)
tlist2, deltat2 = np.linspace(0, endtime2, steps2, retstep=True)
rank=1
rank2=1
ntraj = 1001
run_lowrank=True
run_exact=True
run_mc=True
# run_exact=False
# run_mc=False