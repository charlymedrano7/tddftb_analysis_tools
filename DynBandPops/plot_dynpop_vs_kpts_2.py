# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
import os
import sys
import subprocess

bandfile = open('band.out', 'r')
lines = bandfile.readlines()

fermiline = subprocess.Popen('grep "Fermi level" detailed.out',stdout=subprocess.PIPE,shell=True).communicate()
fermi=float(fermiline[0].split()[4])
idxs = [i for i in range(len(lines)) if 'KPT' in lines[i]]

nstates = idxs[1]-idxs[0]-2
nkpt = len(idxs)                  #la mitad de los kpoints del input
kpts = np.arange(0,nkpt)

bands = np.zeros((nkpt, nstates))
kweights = np.zeros(nkpt)

for ik,idx in enumerate(idxs):
    kweights[ik] = float(lines[idx].split()[5])
    bands[ik,:] = np.array([float(lines[idx + ist].split()[1]) for ist in range(nstates)])
bands = bands - fermi

def plotbandst(ax, kpts, bands):             #función para plotear las bandas 
    for ist in range(nstates):
        ax.plot(kpts, bands[:,ist], c='K')
        ax.plot(-kpts, bands[:,ist], c='K')  #Esto para que se vean las bandas completas y no la mitad
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-kpts.max(), kpts.max())     #


work_dir = os.getcwd() #

laserenergy= work_dir+'/lol/'

molpopuls = [file for file in os.listdir(laserenergy) if "molpopul" in file]
molpopuls.sort()

datapops = np.genfromtxt(laserenergy+molpopuls[0])
time = datapops[:,0]

pops = np.zeros((time.size, nkpt, nstates))
for ik,file in enumerate(molpopuls):
    datapops = np.genfromtxt(laserenergy+file)
    pops[:,ik,:] = datapops[:,1:]

frames = int(sys.argv[1])
timeidxs = []

for i in range(frames):
	timeidxs.append(int(sys.argv[i+2]))

print(timeidxs)

fig, ax = plt.subplots(1, 3, figsize=(18,5), sharex=True, sharey=True)
cols = ['blue', 'red']
for it,t in enumerate(timeidxs):
    plotbandst(ax[it], kpts, bands)                      # BS plot
    ax[it].set_title('time = {:.1f} fs'.format(time[t])) # set title as time frame
    for ist in range(nstates):
        for ik in range(len(kpts)):
            deltapop = (pops[t,ik,ist]-pops[0,ik,ist])*1.0e5    #qué mierda es este factor?
            coloridx = int(np.sign(deltapop)+1.0)//2
            ax[it].scatter(ik, bands[ik,ist], s=np.sqrt(abs(deltapop)), color=cols[coloridx], edgecolor='k')
            ax[it].scatter(-ik, bands[ik,ist], s=np.sqrt(abs(deltapop)), color=cols[coloridx], edgecolor='k') 
    ax[it].set_xticks([-kpts.max(), kpts.max(), 2*kpts.size]);
    ax[it].set_xticklabels(['Z', 'G', 'Z'], fontsize=14);
ax[0].set_ylabel('Energy (eV)',fontsize=16);

plt.savefig('dynbandpops.png', fmt='png', dpi=300, bbox_inches='tight')

