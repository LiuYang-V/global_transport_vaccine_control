## moudule of metapopulation epidemic, plain moudule means no commuting and no transport, 
### this moudule is used to test the L-BFGS-B algorithm
### the variables are transport control and vaccination rate 
### only calculate one day change in epidemic

import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
import time


class Epidemic_solution_p:
    
    def __init__(self, his_data, state, C_ne, A_ne_i, A_ne_d, params, T, beta, delta, c_code, C_R):
        self.N_p = his_data['N_p']     # population size
        #self.N_0 = his_data['N_0']      # initial pandemic data
        self.sigma = his_data['sigma']  # commuting rate
        self.X = his_data['X']          # transportation matrix (dict not matrix)
        self.kpa = his_data['kpa']      # regional daily contact rate
        
        self.N = state
        # neighbor node of transport
        self.C_ne = C_ne         # neighbor node of commute
        self.A_ne_i = A_ne_i     # neighbor node of international air transport
        self.A_ne_d = A_ne_d     # neighbor node of domenstic air transport
        
        self.v = params['v']      # infection period
        self.xi = params['xi']    # latent period


        self.gamma_s = params['gamma_s']
        self.gamma_v = params['gamma_v']
        

        self.c_i = params['c_i']          # {j:c_j^I} infection cost

        self.T = T                        # research period
        
        self.beta = beta
        self.delta = delta # transportation control variable
        self.c_code = c_code # country code 
        self.C_R = C_R      # country region dict

        
    
    
    
    def SD(self, j, N):# the state dynamic calculation
        lam_s = self.kpa[j]*self.gamma_s*N[j][3]/self.N_p[j]
        lam_v = self.kpa[j]*self.gamma_v*N[j][3]/self.N_p[j]
        #beta_j = self.beta[j]
        Delta_j = np.array([[-lam_s,0, 0, 0, 0],\
                            [0, -lam_v, 0, 0, 0],\
                            [lam_s, lam_v, -self.xi, 0, 0,],\
                            [0, 0, self.xi, -self.v, 0],\
                            [0, 0, 0, self.v, 0]])
        n_j = np.array(N[j])
        n_j_new = Delta_j@n_j + n_j
        
        return n_j_new.tolist()
    
    
    
    def get_res(self):
        epi_N = self.N
        cost = 0 # long distance ctl cost, infection cost, commute ctl cost
            
            
        for day in range(self.T):

            epi_N_new = {}
            for j in self.C_R[self.c_code]:
                epi_N_new[j] = self.SD(j, epi_N)

            for j in self.C_R[self.c_code]:
                cost += self.c_i[j]*1000/self.N_p[j]*(epi_N_new[j][3])**2
            
            epi_N = epi_N_new

        return epi_N, cost
    