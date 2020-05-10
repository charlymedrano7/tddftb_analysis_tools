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
timeidxs = [50, 200, 300]         #list of time index to be plotted
circle_size_factor = 1.0e6        #factor for the circles, needed to delta_pops be in the order of 10^3

bandfile = open(root_bands+'band.out', 'r')                                    #open and read the band.out
lines = bandfile.readlines()
fermiline = subprocess.Popen('grep "Fermi level" '+root_bands+'detailed.out',  #finding the Fermi energy
                             stdout=subprocess.PIPE,shell=True).communicate()
fermi=float(fermiline[0].split()[4])                                           #Fermi energy level
idxs = [i for i in range(len(lines)) if 'KPT' in lines[i]]                     #indexs line for each Kpoint

nstates = idxs[1]-idxs[0]-2                 #the number of states in each kpoint
nkpt = len(idxs)                            #the number of indxs is the number of kpoints
                                            #CAREFUL! nkpt is kpt+1 due to the output band.out
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

fermiidxatgamma = np.argmin(abs(bands[nkpt//2,:]))       #Fermi index at gamma point CAREFUL
                                                         #nkpt is kpts+1, where is actually the gamma point?
                                                         #could be nkpt//2 or (nkpt//2)+1
                                                         #resolved: kpoints is equal to zero at index nkpt//2

#Define the range to be studied:
#centered on the fermi level at gamma point
#nplotstates/2 upper and nplotstates/2 lower
states_range = list(range(fermiidxatgamma-nplotstates//2, fermiidxatgamma+nplotstates//2))

np.argmin(abs(kpts))

"""
<plotbandst function> function to plot the band structure automatically
it needs: ax, kpts, the bands info, the range
"""
def plotbandst(ax, kpts, bands, statesRange=None):       
    try:
        plotst = len(statesRange)
    except:
        statesRange = list(range(bands.shape[1]))
    for ist in statesRange:
        ax.plot(kpts, bands[:,ist], c='K', alpha=0.3)
    ax.set_ylim(bands[len(kpts)//2, statesRange[0]], bands[len(kpts)//2, statesRange[-1]])
    ax.set_xlim(-kpts.max(), kpts.max())     #


fig, ax = plt.subplots(1, 1)
plotbandst(ax, kpts, bands, states_range)

work_dir = os.getcwd()

#list of strings cotaining the populations file names sorted
molpopuls = [file for file in os.listdir(root_pops) if "molpopul" in file]
molpopuls.sort()

#datapops contains the data of the first kpoint
datapops = np.genfromtxt(root_pops+molpopuls[0])
time = datapops[:,0]

pops = np.zeros((time.size, nkpt, nstates))
for ik,file in enumerate(molpopuls):
    datapops = np.genfromtxt(root_pops+file)
    pops[:,ik,:] = datapops[:,1:]

nkpt_pops = len(molpopuls)
if (nkpt_pops == nkpt//2):
    print('Considering same spacing in kpoint grid, but folded by simmetry in the interval (0, 0.5)')
elif (nkpt_pops == nkpt//4):
    print('Considering double spacing in kpoint grid for TD run, folded by symmetry in the interval (0, 0.5)')
else:
    print('No clear relationship between kpoint grids used for band structure and TD run.')
    sys.exit()

kpts_pops = np.linspace(0, kpts[-2], nkpt_pops)

fig, axx = plt.subplots(1, len(timeidxs), figsize=(6*len(timeidxs),5), sharex=True, sharey=True)
try:
    len(axx) > 1.0
    ax = axx
except:
    ax = [axx]
cols = ['blue', 'red']
for it,t in enumerate(timeidxs):
    plotbandst(ax[it], kpts, bands, states_range)                      # BS plot
    ax[it].set_title('time = {:.1f} fs'.format(time[t])) # set title as time frame
    for ist in states_range:
        for ik, kpt in enumerate(kpts_pops):
            ikb = np.argmin(abs(kpts-kpt))
            ikb2 = np.argmin(abs(kpts+kpt)) # -(-ik)
            deltapop = (pops[t,ik,ist]-pops[0,ik,ist])*circle_size_factor   #multiplied by circle factor        
            coloridx = int(np.sign(deltapop)+1.0)//2
            ax[it].scatter(kpts[ikb], bands[ikb,ist], s=np.sqrt(abs(deltapop)), color=cols[coloridx], edgecolor='k')
            ax[it].scatter(kpts[ikb2], bands[ikb2,ist], s=np.sqrt(abs(deltapop)), color=cols[coloridx], edgecolor='k') 
    ax[it].set_xticks([kpts.min(), 0, kpts.max()]);
    ax[it].set_xticklabels(['Z', 'G', 'Z'], fontsize=14);
ax[0].set_ylabel('Energy (eV)',fontsize=16);

plt.savefig('dynbandpops.png', fmt='png', dpi=300, bbox_inches='tight')








