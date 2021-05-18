#!/usr/bin/env python3
#------------------------------------------------------------------------------#
#  DFTB+: general package for performing fast atomistic simulations            #
#  Copyright (C) 2006 - 2020  DFTB+ developers group                           #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#
#
'''
Lorentzian spectrum of DFTB+ TD-Casida excitations data
'''

import sys
import optparse
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

USAGE = """usage: %prog -d xx

Reads output from TD Casida calculation and produces absorption spectra.

Needs EXC.DAT file in working directory."""

#plt.rcParams.update({'font.size': 22})


def main():
	parser = optparse.OptionParser(usage=USAGE)
	parser.add_option("-d", "--damping", action="store", dest="tau",
                      help="damping constant in fs, a typical value is <20> (it comes from the FT of the real-time-TDFTB). It is related to the Gamma of the Lorentzian")
	parser.add_option("-i", "--initial", action="store", dest="e_i",
                      help="initial value of energy for the plot (in eV)")
	parser.add_option("-f", "--final", action="store", dest="e_f",
                      help="final value of energy for the plot (in eV)")
	parser.add_option("-n", "--npoints", action="store", dest="points",
                      help="number of points between the initial and final energies (integer)")

	(options, args) = parser.parse_args()

	if len(args) != 0:
		parser.error("You must specify exactly 4 arguments, the damping constant in eV, the desired \
                     energies range to be ploted () and the number of points")
 
	################
	hplanck = constants.physical_constants['Planck constant in eV s'][0]
	c = constants.physical_constants['speed of light in vacuum'][0]
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


	#Factor to convert from femtosecods to electronVolts 
	#(needed to relate gamma with tau)
	#If you are calculating the same system with the same tau in real-time you have to obtain
	#the same curve in Casida (it works and it is checked for singlets)
	factor_fs_to_eV = hplanck*1e15

	print('Calculating spectrum between '+str(options.e_i)+' and '+str(options.e_f)+' eV')
	# Width Gamma of the Lorentzian, related to tau
	gamma = 1./(np.pi*float(options.tau))*factor_fs_to_eV

	energies, wavelengths, spectrum = get_spectrum('./EXC.DAT', float(options.e_i), float(options.e_f),float(options.points), gamma)
	
	#### SPECTRA OUTPUT in eV and nm ####
	np.savetxt('spec-Cas-ev.dat',np.column_stack((energies, spectrum)))
	np.savetxt('spec-Cas-nm.dat',np.column_stack((wavelengths, spectrum)))


if __name__ == "__main__":
	main()
