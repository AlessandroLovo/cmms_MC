#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:57:10 2020

@author: alessandro
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import optimize as op
import uncertainties as unc
import os
import pandas as pd
from PIL import Image

class Spin:
    def __init__(self,value):
        self.value = int(value)
    
    def flip(self):
        self.value *= -1
        return

class Ising:
    
    def __init__(self,dimension,N_spins,boundary='periodic'):
        '''
        D dimensional lattice of spins
        N_spins controls the number of spin on each axis
        boundary:
            'periodic': the system self replicates
            'free': nothing beyond the surface of the system
            1 or -1: the system is surrounded by all aligned spins
        Spins are randomly initialized
        '''
        
        self.D = dimension
        if type(N_spins) == int:
            self.N_spins = np.ones(self.D,dtype=int)*N_spins
        elif len(N_spins) == self.D:
            self.N_spins = np.array(N_spins,dtype=int)
        else:
            raise TypeError('N_spins must be integer or D long list of integers')
            
        self.boundary = boundary
        
        self.spins = []
        for j in range(np.product(self.N_spins + 2)):
            self.spins.append(Spin(2*int(np.random.uniform(0,2)) - 1))
        self.spins = np.array(self.spins)
        
        self.make_surface()
        
        self.free_index_list = np.array([j for j in range(np.product(self.N_spins + 2)) if not self.is_surface(j)])
        if len(self.free_index_list) != np.product(self.N_spins):
            raise ValueError('Surface not correctly computed')
            
        self.N_free_spins = len(self.free_index_list)
        
        self.magnetization = 0
        self.compute_magnetization()
        
    def compute_magnetization(self):
        self.magnetization = np.sum([s.value for s in self.spins[self.free_index_list]])
        
    def average_magnetization(self):
        return self.magnetization/self.N_free_spins
    
    def domain_wall_density(self): # computes the fraction of spins not completely sorrounded by spins with the same value
        ndw = 0
        for j in self.free_index_list:
            s = self.spins[j].value
            indexs = self.get_indexs(j)
            dw = False
            for i in range(self.D):
                indexs[i] += 1
                if s != self.spins[self.get_index(indexs)].value:
                    dw = True
                    break
                indexs[i] -= 2
                if s != self.spins[self.get_index(indexs)].value:
                    dw = True
                    break
                indexs[i] += 1
            if dw:
                ndw += 1
        
        return ndw/self.N_free_spins
    
    
    def get_index(self,indexs):
        if len(indexs) != self.D:
            raise IndexError('Wrong number of indexs')
   
        j = 0
        for i,N in enumerate(self.N_spins):
            if i == 0:
                j += indexs[0]
            else:
                j += np.product((self.N_spins + 2)[:i])*indexs[i]
        return j
    
    def get_indexs(self,index):
        indexs = np.zeros(self.D,dtype=int)
        r = index
        for i in range(1,self.D):
            d = np.product((self.N_spins + 2)[:-i])
            indexs[-i] = int(r/d)
            r -= indexs[-i]*d
        indexs[0] = r
        return indexs
    
    def is_surface(self,j):
        indexs = self.get_indexs(j)
        if 0 in indexs:
            return True
        for i in range(self.D):
            if indexs[i] == self.N_spins[i] + 1:
                return True
    
    def make_surface(self):            
        for j in range(len(self.spins)):
            if self.is_surface(j):
                if self.boundary == 'free':
                    self.spins[j].value = 0
                elif self.boundary == 'periodic':
                    indexs = self.get_indexs(j)
                    for i,k in enumerate(indexs):
                        if k == 0:
                            indexs[i] = self.N_spins[i]
                        elif k == self.N_spins[i] + 1:
                            indexs[i] = 1
                    self.spins[j] = self.spins[self.get_index(indexs)]
                    
                else:
                    self.spins[j].value = int(self.boundary)
                    
        
    def view(self):
        if self.D == 2:
            return Image.fromarray((np.uint8([s.value for s in self.spins]).reshape(self.N_spins + 2) + 1)*127)
    
    
    def save(self,folder,overwrite=False):
        folder = folder.rstrip('/')
        if os.path.exists(folder):
            if not overwrite:
                print('Cannot save: file exists')
                return False
        else:
            os.mkdir(folder)
        
        file = open(folder+'/settings.txt','w')
        s = ''
        for N in self.N_spins:
            s += str(N) + ' '
        s += '\n'
        s += str(self.boundary) + '\n'
        file.write(s)
        file.close()
        
        np.save(folder+'/spins.npy',np.array([s.value for s in self.spins]))
        return True
    
    def copy(self):
        I = Ising(dimension=self.D,N_spins=self.N_spins,boundary=self.boundary)
        for j in self.free_index_list:
            I.spins[j].value = self.spins[j].value
            
        return I
        
    
    
def load_Ising(folder):
    folder = folder.rstrip('/')
    file = open(folder+'/settings.txt','r')
    s = file.readline().rstrip('\n').rstrip(' ')
    N_spins = np.array(s.split(' '),dtype=int)
    boundary = file.readline().rstrip('\n')
    
    I = Ising(dimension=len(N_spins),N_spins=N_spins,boundary=boundary)
    
    spins = np.load(folder+'/spins.npy')
    for j in I.free_index_list:
        I.spins[j].value = spins[j]
        
    return I


def evolve(obj,t_max,k,h=0,criterion='random',check_time=1,logfile='',log_time=100,max_updates=10):
    '''
    Evolves the Ising instance obj for t_max steps using MC metropolis.
    The Hamiltonian of the system is considered to be
    
    E = -J sum_{i nn j} S_i S_j - H sum_{i} S_i
    
    k = J/k_BT, h = H/k_BT
    
    
    criterion:
        'random': at every timestep a random spin is flipped
        'cycle': the spins are flipped in sequence
    
    check_time: number of timesteps every which compute the variation of the energy and check whether to accept the perturbation
    every log_time timesteps saves the total energy, specific energy and the magnetization
    
    log_time must be a multiple of check_time
    '''
    
    if log_time % check_time != 0:
        raise ValueError('log_time must be a multiple of check_time')
    
    I = obj.copy()
    df = pd.DataFrame(data=[],columns=['step','tot_E','spec_E','m','acceptance','wall_density'])
    acceptance_history = np.zeros(log_time//check_time) # 0: rejected, 1: accepted
    
    #dummy I to autocomplete function
    # remember to remove
    #I = Ising(2,3)
    
    
    total_energy = 0
    j_list = []
    j = 0
    w = 1
    delta_E = 0
    delta_m = 0
    acceptance_history[0] = 1
    
    for t in tqdm(range(t_max)):
        if t % check_time == 0 and t != 0:
            #check if the move is good
            accepted = False
            if delta_E <= 0:
                accepted = True
            else:
                r = np.random.uniform(0,1)
                if r < np.exp(-delta_E):
                    accepted = True
            
            if accepted:
                I.magnetization += delta_m
                total_energy += delta_E
                acceptance_history[w] = 1
            else:
                #revert to original situation
                for q in j_list:
                    I.spins[q].flip()
                acceptance_history[w] = 0
            
            j_list = []
            delta_E = 0
            delta_m = 0
            w += 1
        
        if t % (check_time*max_updates) == 0:
            total_energy = 0
            for j in I.free_index_list:
                indexs = I.get_indexs(j)
                spin_sum = 0
                for i in range(I.D):
                    indexs[i] += 1
                    spin_sum += I.spins[I.get_index(indexs)].value
                    indexs[i] -= 2
                    spin_sum += I.spins[I.get_index(indexs)].value
                    indexs[i] += 1
                total_energy += -I.spins[j].value*(h + k*spin_sum)
            I.compute_magnetization()
            
        if t % log_time == 0:
            w = 0
            df.loc[len(df)] = [t, total_energy, total_energy/(k*I.N_free_spins*2*I.D),
                   I.average_magnetization(),np.mean(acceptance_history),I.domain_wall_density()]
            
        
        
        if criterion == 'cycle':
            j = I.free_index_list[t % I.N_free_spins]
        elif criterion == 'random':
            j = I.free_index_list[int(np.random.uniform(0,I.N_free_spins))]
            
        j_list.append(j)
        I.spins[j].flip()
        delta_m += 2*I.spins[j].value
        indexs = I.get_indexs(j)
        spin_sum = 0
        for i in range(I.D):
            indexs[i] += 1
            spin_sum += I.spins[I.get_index(indexs)].value
            indexs[i] -= 2
            spin_sum += I.spins[I.get_index(indexs)].value
            indexs[i] += 1
        delta_E += -2*I.spins[j].value*(h + k*spin_sum)
        
    if logfile != '':
        df.to_csv(logfile,index=False)
        
    return I, df
            
            
        
    
        
    
          
        
    
    
        