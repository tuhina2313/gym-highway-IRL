import numpy as np
import gridworld as gw

def value_iteration(nstates, nactions, T, R, discount = 0.9):
    #Takes in a discount and grid world m with defined Ts, Rs
    #Outputs V(s) vector 
    


    eps = 0.001
    V = np.zeros(nstates)

    optimal_policy = np.zeros(nstates, dtype = int)
    difference = float("inf")
    best_action = 1
    while difference > eps:
        difference = 0
        for s_index in range(nstates):
            reward = R[s_index] #here i is the state index
            tsum = 0
            q_max = float("-inf")
            for a_index in range(nactions):
                    transition_prob = T[s_index,a_index, :]
                    temp = np.dot(transition_prob, R + discount*V)
                    tsum = max(tsum , temp)
                    if q_max < tsum:
                        q_max = tsum
                        best_action = a_index
            
            new_difference = abs(V[s_index] - tsum)
            if new_difference > difference:
                difference = new_difference    
            
            V[s_index] = tsum
            optimal_policy[s_index] = best_action
    return V , optimal_policy