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

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import sys
import subprocess

root_bands='data_dalma/bands_gs/' #path to band.out and detailed.out files
root_pops='4.17/pob/'             #path to the molpopuls.dat files
nplotstates = 50                  #number of bands to be plotted
timeidxs = [50, 200, 300]         #list of time index to be plotted (change for something more user friendly)
circle_size_factor = 1.0e6        #factor for the circles, needed to delta_pops be in the order of 10^3

bandfile = open(root_bands+'band.out', 'r')                                    #open and read the band.out
lines = bandfile.readlines()
fermiline = subprocess.Popen('grep "Fermi level" '+root_bands+'detailed.out',  #finding the Fermi energy
                             stdout=subprocess.PIPE,shell=True).communicate()
fermi=float(fermiline[0].split()[4])                                           #Fermi energy level
idxs = [i for i in range(len(lines)) if 'KPT' in lines[i]]                     #indexs line for each Kpoint

nstates = idxs[1]-idxs[0]-2                 #the number of states in each kpoint
nkpt = len(idxs)                            #the number of indxs is the number of kpoints
                                            #CAREFUL! nkpt is kpoints+1 due to the output band.out
kpts = np.arange(0,nkpt)/(nkpt-1) - 0.5     #kpoints

bands = np.zeros((nkpt, nstates),dtype=float)            #bands array 
kweights = np.zeros(nkpt,dtype=float)                    #kweights

for ik,idx in enumerate(idxs):
    kweights[ik] = float(lines[idx].split()[5])               #extracting kweights from band.out lines
    for ist in range(nstates):
        bands[ik,ist] = np.array([float(lines[idx+ist+1].split()[1])]) #(+1 cause the output has an empty line)
                                                              #extracting all the states energies for each
                                                              #kpoint 
bands = bands - fermi                                         #set the zero point at the Fermi level

# +
try:
    idx = np.where(kpts == 0.0)[0]
    gammaidx = idx[0]
    fermiidxatgamma = np.argmin(abs(bands[gammaidx,:]))       #Fermi index at gamma point

    if (len(idx) > 1):
        sys.exit("More than one k-point with zero coordinates, something fishy")

except:
    sys.exit("No k-point found with zero coordinates, gamma point missing?")
# -

#define the range to be studied:
#centered on the fermi level at gamma point
#nplotstates/2 upper and nplotstates/2 lower
states_range = list(range(fermiidxatgamma-nplotstates//2, fermiidxatgamma+nplotstates//2))

"""
<plotbandst function> function to plot the band structure automatically inside a subplots approach
needs: ax (plt axis objetct), kpts (array containing the kpoints), bands (array of shape (kpoints,nstates)) 
info and statesRange (states range to be plotted).
returns: a plot showing the band structure in the specified range. If the range is not specified the function
will plot all the states by default.
"""
def plotbandst(ax, kpts, bands, statesRange=None):       
    try:
        plotst = len(statesRange)                    #try to use the statesRange when is provided
    except:                                          #if not: the statesRange is all the states by
        statesRange = list(range(bands.shape[1]))    #default (will plot all the states)
    for ist in statesRange:
        ax.plot(kpts, bands[:,ist], c='K', alpha=0.3)
    ax.set_ylim(bands[len(kpts)//2, statesRange[0]], bands[len(kpts)//2, statesRange[-1]])
    ax.set_xlim(-kpts.max(), kpts.max())             #setting plot limits


fig, ax = plt.subplots(1, 1)
plotbandst(ax, kpts, bands, states_range)

work_dir = os.getcwd()

#list of strings cotaining the populations file names sorted
molpopuls = [file for file in os.listdir(root_pops) if "molpopul" in file]
molpopuls.sort()

#datapops contains the data of the first kpoint, this is loaded in advance just to get the size of the arrays below
datapops = np.genfromtxt(root_pops+molpopuls[0])
time = datapops[:,0]

pops = np.zeros((time.size, nkpt, nstates))
for ik,file in enumerate(molpopuls):
    datapops = np.genfromtxt(root_pops+file)       #redefine datapops each step
    pops[:,ik,:] = datapops[:,1:]                  #constructing pops

nkpt_pops = len(molpopuls)
if (nkpt_pops == nkpt//2):
    print('Considering same spacing in kpoint grid, but folded by simmetry in the interval (0, 0.5)')
elif (nkpt_pops == nkpt//4):
    print('Considering double spacing in kpoint grid for TD run, folded by symmetry in the interval (0, 0.5)')
else:
    print('No clear relationship between kpoint grids used for band structure and TD run.')
    sys.exit()

#defining the kpts for the populations calculations
#(usually different from the kpoints used for band calculations)
kpts_pops = np.linspace(0, kpts[-2], nkpt_pops)

fig, axx = plt.subplots(1, len(timeidxs), figsize=(6*len(timeidxs),5), sharex=True, sharey=True)
try:                        #just for plt.subplots
    len(axx) #> 1.0         #if axx is a list (cause len(timeidxs) is grather than 1) can proceed
    ax = axx                  
except:                     #if not, ax is a list of one element containing axx
    ax = [axx]
cols = ['blue', 'red']      #for pops ploting: blue means depopulating, red populating
for it,t in enumerate(timeidxs):                                  
    plotbandst(ax[it], kpts, bands, states_range)             #call the plotbandst function for each time index
    ax[it].set_title('time = {:.1f} fs'.format(time[t]))      #set title as time frame
    for ist in states_range:                                  #
        for ik, kpt in enumerate(kpts_pops):                  #BLOCK FOR THE POPS SCATTER PLOTTING
            
            ikb = np.argmin(abs(kpts-kpt))    #below Fermi: find the nearest kpoint index in bands corresponding
                                              #with the kpoint used in pops (needed to scatter plot then)
            ikb2 = np.argmin(abs(kpts+kpt))   #the same above Fermi.
            deltapop = (pops[t,ik,ist]-pops[0,ik,ist])*circle_size_factor #calculate corresponding delta pop for
                                                                          #the time index, kpoint and state
                                                                          #times by circle factor         
            coloridx = int(np.sign(deltapop)+1.0)//2          #coloridx for pops plot based on the delta sign
            #ploting the pops below Fermi
            ax[it].scatter(kpts[ikb], bands[ikb,ist], s=np.sqrt(abs(deltapop)), color=cols[coloridx], edgecolor='k')
            #Ploting the pops above Fermi
            ax[it].scatter(kpts[ikb2], bands[ikb2,ist], s=np.sqrt(abs(deltapop)), color=cols[coloridx], edgecolor='k') 
    ax[it].set_xticks([kpts.min(), 0, kpts.max()]);
    ax[it].set_xticklabels(['Z', 'G', 'Z'], fontsize=14);
ax[0].set_ylabel('Energy (eV)',fontsize=16);

plt.savefig('dynbandpops.png', fmt='png', dpi=300, bbox_inches='tight')

