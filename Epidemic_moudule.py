## moudule of metapopulation epidemic 
### this moudule is used to test the L-BFGS-B algorithm
### the variables are transport control and vaccination rate 
### only calculate one day change in epidemic

import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
import time


class Epidemic_solution:
    
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
        self.tau = params['tau']  # return rate of daily commuters
        self.eta = params['eta']      # coefficient to change the daily contact
        self.t_s_dom = params['t_s_dom']  # period of stay for domestic travelers
        self.t_s_int = params['t_s_int']  # period of stay for international travelers
        self.gamma_s = params['gamma_s']
        self.gamma_v = params['gamma_v']
        
        self.c_b = params['c_b']          # {j:c_j^b} vaccination cost
        self.c_i = params['c_i']          # {j:c_j^I} infection cost
        self.c_d_d = params['c_d_d']          # {j:c^delta} #cost of transportation control
        self.c_d_i = params['c_d_i']
        self.T = T                        # research period
        
        self.beta = beta
        self.delta = delta # transportation control variable
        self.c_code = c_code # country code 
        self.C_R = C_R      # country region dict
        
        self.init()
        
        
    def init(self):
        self.alpha = {j:0.5 for j in self.N_p}
        
        self.kpa_hat = {}
        for j in self.C_R[self.c_code]:
            X_j_dom = sum(self.X[(j,i)]*self.delta[self.c_code] for i in  self.A_ne_d[j])
            
            X_j_int = 0 # count all international passengers
            for c in self.A_ne_i[j]:
                X_j_int += sum(self.X[(j,i)]*self.delta[c] for i in self.A_ne_i[j][c])
            
            self.kpa_hat[j] = self.kpa[j]+(self.eta-1)*self.alpha[j]*self.kpa[j]*(self.t_s_dom*X_j_dom + self.t_s_int*X_j_int)/self.N_p[j]
                    
        self.sigma_sum = {j: sum(self.sigma[(j,i)] for i in self.C_ne[j]) for j in self.C_R[self.c_code]}
        
        # the transport volume multiplied with control coefficient
        self.X_d = {}
        for j in self.C_R[self.c_code]:
            for i in self.A_ne_d[j]:
                self.X_d[(i,j)] = self.X[(i,j)]*self.delta[self.c_code]
                self.X_d[(j,i)] = self.X[(j,i)]*self.delta[self.c_code]
            for c in self.A_ne_i[j]:
                for i in  self.A_ne_i[j][c]:
                    self.X_d[(i,j)] = self.X[(i,j)]*self.delta[c]
                    self.X_d[(j,i)] = self.X[(j,i)]*self.delta[c]
                    
        # get the epidemic movement X_ij^S ...
        self.X_epi = {}
        for l in self.X_d:
            self.X_epi[l] = [self.X_d[l]*self.N[l[0]][0]/self.N_p[l[0]], self.X_d[l]*self.N[l[0]][1]/self.N_p[l[0]], self.X_d[l]*self.N[l[0]][2]/self.N_p[l[0]], self.X_d[l]*self.N[l[0]][3]/self.N_p[l[0]], self.X_d[l]*self.N[l[0]][4]/self.N_p[l[0]] ]

    
    
    def Lam(self, j, N):#  j is subpopulation area code, N is the population state at time t
        N_star = self.N_p[j]
        #kpa_j = self.kpa_hat[j][t]
        
        lam_jj_s = self.kpa_hat[j]*self.gamma_s/self.N_p[j] * (N[j][3]/(1+self.sigma_sum[j]/self.tau) + sum(N[i][3]*self.sigma[(i,j)]/(self.tau+self.sigma_sum[i])  for i in self.C_ne[j]))/(1 + self.sigma_sum[j]/self.tau) 
        lam_jj_v = self.kpa_hat[j]*self.gamma_v/self.N_p[j] * (N[j][3]/(1+self.sigma_sum[j]/self.tau) + sum(N[i][3]*self.sigma[(i,j)]/(self.tau+self.sigma_sum[i])  for i in self.C_ne[j]))/(1 + self.sigma_sum[j]/self.tau) 
        
        lam_ji_s = sum( self.sigma[(j,i)]* self.kpa_hat[i]* self.gamma_s/self.N_p[i]* (N[i][3]/(1+self.sigma_sum[i]/self.tau) + sum(N[l][3]*self.sigma[(l,i)]/(self.tau+self.sigma_sum[l])  for l in self.C_ne[i] ))  for i in self.C_ne[j] )/(self.tau + self.sigma_sum[j])
        
        lam_ji_v = sum( self.sigma[(j,i)]* self.kpa_hat[i]* self.gamma_v/self.N_p[i]* (N[i][3]/(1+self.sigma_sum[i]/self.tau) + sum(N[l][3]*self.sigma[(l,i)]/(self.tau+self.sigma_sum[l])  for l in self.C_ne[i] )) for i in self.C_ne[j] )/(self.tau + self.sigma_sum[j])
        
        lam_j_s = lam_jj_s + lam_ji_s
        lam_j_v = lam_jj_v + lam_ji_v
        return lam_j_s, lam_j_v
    
    
    
    def SD(self, j, N):# the state dynamic calculation
        lam_s, lam_v = self.Lam(j, N)
        beta_j = self.beta[j]
        Delta_j = np.array([[-lam_s-beta_j,0, 0, 0, 0],\
                            [beta_j, -lam_v, 0, 0, 0],\
                            [lam_s, lam_v, -self.xi, 0, 0,],\
                            [0, 0, self.xi, -self.v, 0],\
                            [0, 0, 0, self.v, 0]])
        n_j = np.array(N[j])
        n_j_new = Delta_j@n_j + n_j
        
        X_ij_n = np.array([sum(self.X_epi[(i,j)][a] for i in self.A_ne_d[j]) + sum(sum(self.X_epi[(i,j)][a] for i in self.A_ne_i[j][c]) for c in self.A_ne_i[j]) for a in range(5)])
        
        X_ji_n = np.array([sum(self.X_epi[(j,i)][a] for i in self.A_ne_d[j]) + sum(sum(self.X_epi[(j,i)][a] for i in self.A_ne_i[j][c]) for c in self.A_ne_i[j]) for a in range(5)])
        
        N_j_tp1 = n_j_new + (X_ij_n - X_ji_n)
        return N_j_tp1.tolist()
    
    
    
    def get_res(self):
        epi_N = self.N
        cost = {'lt': 0, 'inf': 0, 'vac': 0} # long distance ctl cost, infection cost, vaccine cost
        
        for day in range(self.T):

            epi_N_new = {}
            for j in self.C_R[self.c_code]:
                epi_N_new[j] = self.SD(j, epi_N)

            for j in self.C_R[self.c_code]:
                cost['inf'] += self.c_i[j]*1000/self.N_p[j]*(epi_N_new[j][3])**2
                cost['vac'] += self.c_b[j]*epi_N[j][0]*self.beta[j]
            
            epi_N = epi_N_new
        
        for j in self.C_R[self.c_code]:
            for i in self.A_ne_d[j]:
                cost['lt'] += self.T*self.c_d_d*self.X[(i,j)]*(1-self.delta[self.c_code])**2
            for c in self.A_ne_i[j]:
                for i in self.A_ne_i[j][c]:
                    cost['lt'] += self.T*self.c_d_i*self.X[(i,j)]*(1-self.delta[c])**2
        return epi_N, sum(cost[n] for n in cost)
    