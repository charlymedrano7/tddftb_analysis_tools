# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sys
import os
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import math
from scipy import constants

# %matplotlib inline
# -

area_ribbon_5 = (21.3129*57.262)*1e-20 #largo de la celda * ancho de H a H
area_ML_TDI = area_ribbon_5


def calc_spec(filename,tau):
    muy = np.loadtxt(filename)#[0:2000]
    length = muy.shape[0]
    damp = np.exp(-muy[:,0]/tau)
    mu = muy[:,2] - muy[0,2]
    time = muy[:,0]
    delt = time[1] - time[0]
    spec = np.fft.rfft(damp*mu, 10*length)
    hplanck = constants.physical_constants['Planck constant in eV s'][0] * 1.0E15
    energsev = np.fft.rfftfreq(10*length, delt) * hplanck
    frec = np.fft.rfftfreq(10*length, delt) * 1.0E-15 
    absorption = -2.0 * energsev * spec.imag / np.pi 
    return energsev, absorption


def calc_spec2(filename,tau,field):
    hbar = 0.65821188926 # in eV fs
    muy = np.loadtxt(filename)
    length = muy.shape[0]
    damp = np.exp(-muy[:,0]/tau)
    mu = (muy[:,2] - muy[0,2])/field
    time = muy[:,0]
    delt = time[1] - time[0]
    spec = np.fft.rfft(damp*mu, 10*length)
    omegas = np.fft.rfftfreq(10*length, delt) * 2 * np.pi * 1.0e15
    omegasfs = np.fft.rfftfreq(10*length, delt) * 2 * np.pi
    alphaim =  time[-1] * spec.imag / float(length)
    cross_section = omegas * (-alphaim) * constants.e * 1.0e-20 / constants.c / constants.epsilon_0
    return omegasfs*hbar, 4*np.pi**3*cross_section


# +
### Esta función calcula el espectro en unidades absolutas a partir de una componente del mu
### y el tiempo (ambos deben ser arrays)
def calc_specOF(time_array, mu_array, tau,field):
    hbar = 0.65821188926 # in eV fs
    muy = mu_array
    time = time_array
    length = muy.shape[0]
    damp = np.exp(-time/tau)
    mu = (muy[:] - muy[0])/field
    delt = time[1] - time[0]
    spec = np.fft.rfft(damp*mu, 10*length)
    omegas = np.fft.rfftfreq(10*length, delt) * 2 * np.pi * 1.0e15
    omegasfs = np.fft.rfftfreq(10*length, delt) * 2 * np.pi
    alphaim =  time[-1] * spec.imag / float(length)
    cross_section = omegas * (-alphaim) * constants.e * 1.0e-20 / constants.c / constants.epsilon_0
    return omegasfs*hbar, 4*np.pi**3*cross_section

### Esta función lee el archivo mu.dat devolviendo todas las componentes en forma de array
def readMu(filename):
    mu = np.loadtxt(filename)
    time = mu[:,0]
    mux = mu[:,1]
    muy = mu[:,2]
    muz = mu[:,3]
    return time, mux, muy, muz

### Esta función corta el time y el mu según un offset y una window. Necesita el parámetro de time step
def muCut(time_array, mu_array, offset, window, time_step): #offset and windows in fs, time_step in atomic units
    t_abs_to_fs = 0.024188843                               #factor from atomic time units to fs
    
    offset_index = int(offset/(time_step*t_abs_to_fs))     #find the offset index requested (+1 cause the zero)
    offset_new = time_array[offset_index]                   #set new offset (the bigger nearest value in data)
    window_index = int(window/(time_step*t_abs_to_fs))    #same for window
    window_new = time_array[window_index]
    
    mu_cut = mu_array[offset_index:offset_index+window_index]       #cuting time and mu
    time_cut = time_array[offset_index:offset_index+window_index]
    
    return time_cut, mu_cut


# -

workdir = "/home/charly/Palma_project/KPOINT/"

# +
# PLOT 1

tau = 10
field = 0.0001

#Ribbon
e_R5, s_R5 = calc_spec2(workdir+'46AGNR5/muy.dat',tau, field)

#Ribbon+TDI
e_R5_TDI, s_R5_TDI = calc_spec2(workdir+'46AGNR5_TDI/muy.dat',tau, field)

# +
offset = 0.0
window = 48.0
t_i = 0.0
t_f = 48.0
step = 48.0
time_step = 0.2 #(atomic units from input)
tau = 10
field = 0.0001

plt.figure(figsize=(8,8))
plt.xlim(0,5)
plt.ylim(-0.2,0.4)

for i in np.arange(t_i,t_f,step):#[:-1]:
    offset += i
    print(offset)
    time, mux, muy, muz = readMu(workdir+'46AGNR5/muy.dat')     #Lee el archivo muy.dat
    time_cut, mu_cut = muCut(time, muy, offset, window, time_step)  #corta el momento dipolar 
    ener, spec = calc_specOF(time_cut, mu_cut, tau,field)
    
    plt.plot(ener,spec/area_ribbon_5, label='fake-probe')

plt.plot(e_R5_TDI,s_R5_TDI/area_ribbon_5, label='normal-spec')
plt.plot(e_R5,s_R5/area_ribbon_5, label='normal-spec')
plt.legend()
    
    
    

# -


