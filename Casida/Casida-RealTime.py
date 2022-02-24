# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

# %matplotlib inline
# -

plt.rcParams.update({'font.size': 22})

hplanck = constants.physical_constants['Planck constant in eV s'][0]
c = constants.physical_constants['speed of light in vacuum'][0]

# +
"""
### <Lorentzian> ###
Function to calculate the Lorentzian function L(x) in a given point
# It needs: Array of x-values <[x]>, interested point in which the Lorentzian
will be adjust <x_0> and the desired width for the peak <gamma>
# Returns: an array of numbers containing the L(x)
"""
def Lorentzian(x, x_0, gamma):
    return (1/np.pi)*((0.5*gamma)/((x-x_0)**2+(0.5*gamma)**2)) #Lorentzian function

"""
### <get_spectrum> ###
Function to obtain the excitations expectrum from the casida output from dftb+
# It needs: the path to the 'EXC.DAT' output file from casida <'casida_out'>, the 
desired initial and final energies (in eV) for the spectrum <e_i> and <e_f>, the number of 
points for the energies array <points> and the lorentzian width parameter <gamma>
# It returns: the arrays of <[energies]>, <[wavelengths]> and <[spectrum]>
"""
def get_spectrum(file, e_i, e_f, points, gamma):
    excitations = np.genfromtxt(file, skip_header=5)[:,:2]  #take first two columns of EXC.DAT
    energies = np.linspace(e_i, e_f, num=points)            #array of energies
    wavelengths = np.zeros_like(energies)                   #array for wavelengths
    wavelengths[1:-1] = (hplanck*c/energies[1:-1])*1e9
    wavelengths[0] = wavelengths[1]                         #just repeat last value ("inf")
    exc = excitations[:,0]                                  #ecitation energies
    osc_str = excitations[:,1]                              #oscilation strength associated
    spectrum = np.zeros_like(energies)                      #array for the spectrum
    for i, energy in enumerate(exc):                        #adjust a lorentzian to each exc. energy
        spectrum[:] += osc_str[i]*Lorentzian(energies, energy, gamma) 
    return energies, wavelengths, spectrum      


# +
### PARAMETERS ###
tau = 20         #from the real time (in fs)
e_i = 0          #initial energy for the energies array    
e_f = 10         #final energy
points = 5000   #number of points in the energies array (defines the resolution)

#Factor to convert from femtosecods to electronVolts 
#(needed to relate gamma with tau)
factor_fs_to_eV = hplanck*1e15

# Width Gamma of the Lorentzian, related to tau
gamma = (1/(np.pi*tau))*factor_fs_to_eV

energies, wavelengths, spectrum = get_spectrum('benzene/casida/EXC.DAT', e_i, e_f, points, gamma)
# -

real_t = np.genfromtxt('benzene/realtime/spec-ev_d20.dat')

#Normalization factor based in the max values of the peaks
normal_fac = real_t[:,1][0:2500].max()/spectrum.max()      #realtime peak divided by casida peak

# +
fig, ax = plt.subplots(1,1, figsize=(8,8), sharex=False, sharey=False)

ax.plot(real_t[:,0], real_t[:,1]/normal_fac, label='RT-TD-DFTB', linewidth=3.0, color='black')
ax.plot(energies, spectrum,'.', label='casida', linewidth=1.0, color='red', alpha=0.8)
ax.legend()

ax.set_xlim([5,9])
ax.set_ylim([0,10])
ax.set_xlabel('Energy (eV)')
ax.set_ylabel('Absorption (arb. units)')

fig.suptitle('Benzene')

plt.savefig('spectrum.png', format='png', dpi=300, transparent=False )
# -


