# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import simps, cumtrapz

from scipy import constants
ev_to_j = constants.physical_constants['electron volt'][0]
uma = constants.physical_constants['atomic unit of mass'][0]
ang = 1.0E-10
#autime = constants.physical_constants['atomic unit of time'][0]
fs = 1.0E-15
ha_to_ev = constants.physical_constants['Hartree energy in eV'][0]
ang_to_bohr = constants.physical_constants['Bohr radius'][0] * 1.0E10
hplanck = constants.physical_constants['Planck constant in eV s'][0] * 1.0E15

# %matplotlib inline
# -

hplanck


# +
def readCoords(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    natoms = int(lines[0].strip())
    coords = np.zeros((natoms, 3))
    names = [0] * natoms
    for i in range(natoms):
        coords[i, :] = [float(x) for x in lines[i+2].strip().split()[1:4]]
        names[i] = lines[i+2].strip().split()[0]
    f.close()
    return names, coords

def plot_struc(x, y, platform):
    for at in range(x.size):
        plt.axis("equal")
        if platform == 'Yes':
            plt.scatter(x[at],y[at],c='k', s=1)
        else:
            plt.scatter(x[at],y[at],c='g', s=1)

"""
coords: array read from file with readCoords, layer: integer, number of atoms of the monolayer,
charges: array read from file qsvst, timearr: time array, times: times to be calculated,
prefix: prefix name for the plot to be saved.
"""
def plotelectricfield(coords, layer, charges, timearr, times, prefix, scale=None):
    #Only for frozen simulations
    res = 0.1                   #resolution in Angstrom
    natoms = coords.shape[0]    
    xx = coords[:,0]
    yy = coords[:,1]
    zz = coords[:,2]
    xmax = 1.1*np.abs(xx).max() #Para que el max de la caja sea mayor que la molécula
    ymax = 1.1*np.abs(yy).max()
    zmax = 1.1*np.abs(zz).max()
    
    domy, domz = np.mgrid[-ymax:ymax:res,-zmax:zmax:res] #definir grilla 2D (mgrid hace la grilla con la res)
    nxpoints = int(2*xmax/res)                            #
    xrange = np.linspace(-xmax, xmax, nxpoints)          
#     print (nypoints)                                      #borrable
    alfa = 0.0                                            #radio de exclusión para el cálculo del E (el E 
                                                          #se va al carajo)
    timeidxs = [np.argmin(abs(timearr-t)) for t in times] #index del time para calc
    xval = xx[0]
    print(xval)
    electfields = []
    for i,timeidx in enumerate(timeidxs):
        elp = np.zeros_like(domy)                         #define electric potential 
#         for xval in xrange:                               #me muevo en y porque lo calculo aquí
        for at in range(natoms-layer, natoms):                      #acá sólo tengo que iterar sobre las moléculas
            elp += (charges[timeidx,at]-charges[0,at])\
            /np.sqrt((domy-yy[at])**2 + (xval-xx[at])**2 + (domz-zz[at])**2 + alfa**2)
        elp *= 9.9e9*1.6e-19*1e10
#         elp /= nxpoints                           #promedio (carga neta del átomo dividido la norma)
#         print(elp.shape)
        grad = np.gradient(elp, res, axis=0)           #gradiente (derivada parcial en y)
        elect_field = grad.sum(axis=1)/grad.shape[1]   #calcula el promedio en z 
        electfields.append(elect_field)                #appendeo los campos en la lista   
        
#Matplotlib
#         if scale is None:
#             z_min, z_max = -np.abs(elp).max(), np.abs(elp).max()
#         else:
#             z_amp = scale
#             z_min, z_max = -z_amp, z_amp

#         plt.figure(num=None, figsize=(5, 4), dpi=200)
#         plt.pcolormesh(domz,domy, elp, vmin=z_min, vmax=z_max, cmap=plt.get_cmap('seismic'))
#         if i == len(timeidxs)-1:
#             plt.colorbar(ticks=[z_min, z_max], format="%.6f")
#         plt.title('$t = {:.1f}$ fs'.format(timearr[timeidx]))
#         plot_struc(zz[:natoms-layer], yy[:natoms-layer], 'Yes')
#         plot_struc(zz[natoms-layer:natoms], yy[natoms-layer:natoms], 'No')
#         plt.axis('off')
#         plt.savefig(prefix+'pot_'+str(i).zfill(3), dpi=150, bbox_inches='tight')
# #         plt.close()
    
    y_array = np.arange(-ymax,ymax,res)
    return y_array, electfields



# HAY QUE CORREGIR EL CAMPO (MENOS EL GRADIENTE)
# PARTIR LA FUNCIÓN EN DOS: UNA PARA CALCULAR EL CAMPO Y OTRA PARA PLOTEAR EL POTENCIAL SOBRE LA PLATAFORMA
# -

# rootdir = '{}/field0.0001/'.format(55)
qdata = np.genfromtxt('qsvst.dat')
names, coords = readCoords('coords.xyz')
charges = qdata[:,2:]
timearr = qdata[:,0]

time = timearr
yy, elect_field_y = plotelectricfield(coords,324, charges, timearr, time, '24_')

yy.shape

# +
len(elect_field_y)

elect_field_y = np.array(elect_field_y)

# +
# plt.plot(yy, elect_field_y[:,0])
# plt.plot(yy, elect_field_y[:,100])

plt.imshow(elect_field_y, aspect='auto')
# -

E0 = 0.001
omega = 1.3/0.658

average_elect_y = elect_field_y.sum(axis=1)/elect_field_y.shape[1]

yy[300]

# +
plt.figure(figsize=(12,8))

plt.plot(timearr, -elect_field_y[:,300], label=r'near field at -1.5 $\AA$', lw=2.0)
plt.plot(timearr, E0*np.sin(omega*timearr), label='external field', lw=2.0)
plt.plot(timearr, -elect_field_y[:,300]+E0*np.sin(omega*timearr), label='near+external', lw=3.0)
# plt.plot(timearr, -average_elect_y, label='average near field')

plt.xlim(0,30)

plt.title('1.3 eV ')
plt.legend(fontsize=20)
plt.xlabel('time (fs)', fontsize=20)
plt.ylabel('Electric field', fontsize=20)
# -

time = [12]
plotelectricfield(coords,324, charges, timearr, time, '12_')

time = [18]
plotelectricfield(coords,324, charges, timearr, time, '00_')

# rootdir = '{}/field0.0001/'.format(55)
qdata_post = np.genfromtxt('/home/charly/Palma_project/KPOINT/ActSpecs_frank/ActSpec_TDI/2.1/qsvst.dat')
names, coords = readCoords('/home/charly/Palma_project/KPOINT/ActSpecs_frank/ActSpec_TDI/2.1/coords.xyz')
charges_post = qdata_post[:,2:]
timearr_post = qdata_post[:,0]

time_post = timearr_post
yy_post, elect_field_y_post = plotelectricfield(coords,324, charges_post, timearr_post, time_post, '24_')

# +
len(elect_field_y_post)

elect_field_y_post = np.array(elect_field_y_post)
# -

plt.imshow(elect_field_y_post, aspect='auto')

E0 = 0.001
omega_post = 2.1/0.658

average_elect_y_post = elect_field_y_post.sum(axis=1)/elect_field_y_post.shape[1]

# +
plt.figure(figsize=(12,8))

plt.plot(timearr_post, -elect_field_y_post[:,300], label=r'near field at -1.5 $\AA$', lw=2.0)
plt.plot(timearr_post, E0*np.sin(omega_post*timearr_post), label='external field', lw=2.0)
plt.plot(timearr_post, -elect_field_y_post[:,300]+E0*np.sin(omega_post*timearr_post), label='near+external', lw=3.0)
# plt.plot(timearr, -average_elect_y, label='average near field')

plt.xlim(0,30)

plt.title('2.1 eV', fontsize=40)
plt.legend(fontsize=20, loc='upper right')
plt.xlabel('time (fs)', fontsize=20)
plt.ylabel('Electric field', fontsize=20)
# -
# rootdir = '{}/field0.0001/'.format(55)
qdata_exc = np.genfromtxt('/home/charly/Palma_project/KPOINT/lasers/tdi/qsvst.dat')
names, coords = readCoords('/home/charly/Palma_project/KPOINT/lasers/tdi/coords.xyz')
charges_exc = qdata_post[:,2:]
timearr_exc = qdata_post[:,0]

time_exc = timearr_exc
yy_exc, elect_field_y_exc = plotelectricfield(coords,324, charges_exc, timearr_exc, time_exc, '24_')

# +
len(elect_field_y_exc)

elect_field_y_exc = np.array(elect_field_y_exc)
# -

plt.imshow(elect_field_y_post, aspect='auto')

E0 = 0.001
omega_exc = 1.79/0.658

average_elect_y_exc = elect_field_y_exc.sum(axis=1)/elect_field_y_exc.shape[1]

# +
plt.figure(figsize=(12,8))

plt.plot(timearr_exc, -elect_field_y_exc[:,300], label=r'near field at -1.5 $\AA$', lw=2.0)
plt.plot(timearr_exc, E0*np.sin(omega_exc*timearr_exc), label='external field', lw=2.0)
plt.plot(timearr_exc, -elect_field_y_exc[:,300]+E0*np.sin(omega_exc*timearr_exc), label='near+external', lw=3.0)
# plt.plot(timearr, -average_elect_y, label='average near field')

plt.xlim(0,30)

plt.title('1.79 eV', fontsize=40)
plt.legend(fontsize=20, loc= "upper right")
plt.xlabel('time (fs)', fontsize=20)
plt.ylabel('Electric field', fontsize=20)
# -






