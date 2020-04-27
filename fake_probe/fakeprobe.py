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
hbar = 0.65821188926
hplanck = hbar * 2 * np.pi
# -

area_ribbon_5 = (21.3129*57.262)*1e-20 #largo de la celda * ancho de H a H
area_ML_TDI = area_ribbon_5


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
    return omegasfs*hbar, 4*np.pi**3*cross_section, time, mu, mu*damp


# +
### This function calculates calculate the abs spectrum in absolute units 
### you need to multiply the output by the area of your system 
### It needs: mu and time (arrays), tau and field (floats) 
### Returns: energies and absorption (arrays)
def calc_specOF(time_array, muy, tau, field):
    hbar = 0.65821188926 # in eV fs
    time = time_array - time_array[0]              #the time needs to start from zero for the damping
    length = muy.shape[0]
    damp = np.exp(-time/tau)
    mu = (muy[:] - muy[0])/field
    delt = time[1] - time[0]
    spec = np.fft.rfft(damp*mu, 10*length)
    omegas = np.fft.rfftfreq(10*length, delt) * 2 * np.pi * 1.0e15
    omegasfs = np.fft.rfftfreq(10*length, delt) * 2 * np.pi
    alphaim =  time[-1] * spec.imag / float(length)
    cross_section = omegas * (-alphaim) * constants.e * 1.0e-20 / constants.c / constants.epsilon_0
    return omegasfs*hbar, 4*np.pi**3*cross_section, spec

### This function reads the file 'mu.dat' to return the columns
### It needs: the pathway where the file 'mu.dat' is
### returns: time, mux, muy and muz (arrays)
def readMu(filename):
    mu = np.loadtxt(filename)
    time = mu[:,0]
    mux = mu[:,1]
    muy = mu[:,2]
    muz = mu[:,3]
    return time, mux, muy, muz

### This function cut the time and mu following 2 parameters of time (offset and window)
### It needs: time and mu (arrays), offset and window (floats) and the time_step (of the simulation a.u.)
### Returns: cutted time and mu (arrays)
def muCut(time_array, mu_array, offset, window, time_step): #offset and windows in fs, time_step in atomic units
    t_abs_to_fs = 0.024188843                               #factor from atomic time units to fs
    
    offset_index = int(offset/(time_step*t_abs_to_fs))      #find the offset index requested (+1 cause the zero)
    offset_new = time_array[offset_index]                   #set new offset (the bigger nearest value in data)
    window_index = int(window/(time_step*t_abs_to_fs))      #same for window
    window_new = time_array[window_index]
    
    mu_cut = mu_array[offset_index:offset_index+window_index]       #cuting time and mu
    time_cut = time_array[offset_index:offset_index+window_index]
    
    return time_cut, mu_cut


# -

workdir = "./"

# +
# PLOT 1

tau = 5
field = 0.0001

#Ribbon
e_R5, s_R5, t_R5, mu_R5, mu_damp_R5 = calc_spec2(workdir+'muy_R.dat',tau, field)

#Ribbon+TDI
e_R5_TDI, s_R5_TDI, t_R5_TDI, mu_R5_TDI, mu_damp_R5_TDI = calc_spec2(workdir+'muy_R+TDI.dat',tau, field)

# +
# PLOT 2

tau = 5
field = 0.0001

#Ribbon
e_R5_, s_R5_, t_R5_, mu_R5_, mu_damp_R5_ = calc_spec2(workdir+'muy_resol_R.dat',tau, field)

#Ribbon+TDI
e_R5_TDI_, s_R5_TDI_, t_R5_TDI_, mu_R5_TDI_, mu_damp_R5_TDI_ = calc_spec2(workdir+'muy_resol_R+TDI.dat',tau, field)
# -

# ### Calibrado
# - Window:
#
# para definir la ventana de tiempo que necesitamos tenemos que pensar en las energías que nos interesa resolver.
# La energía mínima que queremos ver va a definir el $\tau$ (período) que necesitamos segun:
#
# $ \displaystyle E = h\nu = \frac{h}{\tau}$
#
# para 1 eV tendríamos:
#
# $ \displaystyle \tau = \frac{4.14^{-15} eVs}{1 eV} = 4.14 fs $
#
# por el padding que se usa en la FFT suele no ser suficiente con un periódo para el time window. Por lo general
# se usa como $t_w = 4\tau = ~16 fs $ 
#
# - Damping:
#
# El damping aplicado a la ventana tiene que ser tal que el momento dipolar quede cercano a cero al final de la ventana (ya no oscile tanto). Dependerá entonces del ancho de la ventana. Hay que ver los momentos dipolares. Devería caer a cero suavemente al final.

plt.plot(t_R5_TDI, mu_R5_TDI)
plt.plot(t_R5_TDI, mu_damp_R5_TDI)
plt.xlim(0,10)
# +
offset = 0.0   # starting offset 
window = 15.0   # time window to be study (it depends on the energies that we want to study)
t_i = 0.0       # 
t_f = 48.0      # Simulated time
step = 2.35      # see below (hace que no sea necesario rotar)
time_step = 0.2 # time step in atomic units from input 
tau = 5         # for damping
field = 0.0001  # field strength from input

energs = []
alphas = []
specs = []

offsets = [offi for offi in np.arange(offset,t_f,step)]

time, mux, muy, muz = readMu(workdir+'muy_R+TDI.dat')  #read the file muy.dat
#should we add an if to be sure offset < t_f?

for i in np.arange(offset,t_f,step):                    #loop between the offset and the t_f
    offset_calc = i
    if offset_calc+window <= t_f:                       #to be sure not to be outside the simulated time
        print(offset_calc)
        time_cut, mu_cut = muCut(time, muy, offset_calc, window, time_step)  #cut dipole moment
        ener, spec, alpha = calc_specOF(time_cut, mu_cut, tau,field)                #calc spec of this mu_cut
        energs.append(ener)
        alphas.append(alpha)
        specs.append(spec)

    else:                                               #when we are outside of the simulated time
        print('STOPPED')                                 
        print('offset '+'{:.0f}'.format(offset)+' +step '+'{:.0f}'.format(i-offset)+' +window '
              +'{:.0f}'.format(window)+' = '+'{:.0f}'.format(window+i)+
              ' and is outside the simulated time of '+ '{:.0f}'.format(t_f)+'fs')
        break
# -
from matplotlib.pyplot import cm
cols = list(cm.rainbow(np.linspace(0,1,len(energs))))

# +
plt.figure(figsize=(8,8))
plt.xlim(0,5)
plt.ylim(-0.1,0.15)
for isp in range(len(energs)):
    plt.plot(energs[isp],specs[isp]/area_ribbon_5, color=cols[isp], label='offset = {:.2f} fs'.format(offsets[isp]))
plt.title('window length = {:.0f}'.format(window))
plt.plot(e_R5_TDI,s_R5_TDI/area_ribbon_5, 'k:', lw=5, label='R+TDI normal-spec')
plt.plot(e_R5,s_R5/area_ribbon_5, 'k--', label='R normal-spec')
plt.legend(loc='upper right')


# plt.plot(e_R5_TDI_,s_R5_TDI_/area_ribbon_5, 'k', lw=2, color='red', label='resol R+TDI normal-spec')
# plt.plot(e_R5_,s_R5_/area_ribbon_5, 'k--', color='red', label='resol R normal-spec')


# +
# offset is determined to be coincident with the period of the main low energy peak, which modulates
# the absorption vs. time signal. This way we "catch" the dipole signal always at the same part of the oscillation
# ot the main peak at 1.77 eV

period = (1/1.77)*hplanck
print(period)
# -


