from re import X
import numpy as np

def price_xyz(A_t_d, B, t_e, r, sigma, t_d, numsteps, n):

    del_t = np.float(n)/np.float(numsteps)

    j_u = np.exp(sigma*np.sqrt(del_t))
    j_d = np.exp(-sigma*np.sqrt(del_t))
    p_u = (np.exp(r*del_t*0.5)-np.exp(-sigma*np.sqrt(del_t*0.5)))/(np.exp(sigma*np.sqrt(del_t*0.5))-np.exp(-sigma*np.sqrt(del_t*0.5)))
    p_d = (np.exp(sigma*np.sqrt(del_t*0.5))-np.exp(r*del_t*0.5))/(np.exp(sigma*np.sqrt(del_t*0.5))-np.exp(-sigma*np.sqrt(del_t*0.5)))

    A_array = np.empty((numsteps,2*numsteps-1))

    for i in range(numsteps):
        for j in range(2*i+1):
            A_array[i][numsteps-1-i+j] = A_t_d*np.power(j_d,i-j)

    X_array = np.empty_like(A_array)

    for j in range(2*numsteps-1):
        X_array[numsteps-1][j] = np.max([A_array[numsteps-1][j]-B,0])

    for i in reversed(range(numsteps-1)):
        for j in range(2*i+1):
            X_array[i][numsteps-1-i+j] = ((p_u*X_array[i+1][numsteps-i+j])+((1-p_u-p_d)*X_array[i+1][numsteps-1-i+j])+(p_d*X_array[i+1][numsteps-2-i+j]))*np.exp(-r*del_t)

    value = np.empty(numsteps)

    for i in range(numsteps):
        value[i] = X_array[i][numsteps-1]

    X_td = X_array[0][numsteps-1]

    return X_td

print(price_xyz(317.0,297.0,0.0,0.035,0.025,0,1000,462))

delta = 1e-10

a1 = price_xyz(317.0,297.0,0.0,0.035,0.025+delta,0,1000,462)
a2 = price_xyz(317.0,297.0,0.0,0.035,0.025,0,1000,462)
a3 = price_xyz(317.0,297.0,0.0,0.035,0.025-delta,0,1000,462)

print(0.5*(((a1-a2)/delta)+((a2-a3)/delta)))
