#!/usr/bin python3

"""density_on_grid_timeprop_dftb.py: Calculates electronic density difference in real space grid \
   and saves it as CUBE files, from density matrix dumped in time propagations using DFTB+.
   Needs files ptable.csv and wfc file for the SK parameter set, downloadable from dftb.org."""

__author__      = "Franco Bonafe, Cristian G. Sanchez"
__maintainer__  = "Franco Bonafe"
__email__       = "fbonafe@unc.edu.ar"
__copyright__   = "Copyright 2018, FCQ UNC"
__status__      = "Prototype"

import os
from collections import defaultdict
import numpy as np
import csv

from numba import jit   ###njit
# from numba.typed import List
from scipy.special import sph_harm as Ynm #Ynm(m, n, theta, phi), theta=azimuthal, phi=polar
from scipy import constants
import readsto
import concurrent.futures

###### Variables to set before running (EDIT) ############# 


#grid points
nx = 32
ny = 32
nz = 32

iniframe = 0          #initial and final frames (depends on the dftb input)
endframe = 1
frameinterval = 1     #interval of frames to take into account

DUMPBIN_DIR = '../'  # directory where the *dump.bin files are located
CUBES_DIR = './cubes/' # directory where the cubefiles will be stored

coordfile = './geo.xyz'  ## edit accordingly
rhodumpfile = '0ppdump.bin' # *dump.bin file for step = 0 (ground state density), edit

wfc_filename = 'wfc.mio-1-1.hsd'
sto_filename = 'STO.mio.dat'

##########################################################




### Definition of functions #######################

nelemread = 2             #number of elements in the system

orbs_per_l = [1, 3, 5, 7]    #l quantum number for each type of orbital (do not change)

if not os.path.exists(CUBES_DIR):
    os.makedirs(CUBES_DIR)

ptable_data = csv.DictReader(open("ptable.csv")) #ptable.cvs file needed
ptable = {}
for row in ptable_data:
    symbol = row[' symbol'].strip()
    ptable[symbol] = int(row['atomicNumber'])

def getNorbs(atomzs, lmax):
    norbs = 0
    for iat in atomzs:
        for ll in range(lmax[iat]+1):
            norbs += orbs_per_l[ll]
    return norbs

def readCoords(thisfile):
    cfile = open(thisfile, 'r')
    thisfile = cfile.readlines()
    natoms = int(thisfile[0].strip())
    coords = [[],[],[]]
    atomZ = []
    for atom in range(2,natoms+2):
        name = thisfile[atom].strip().split()[0]
        atomZ.append(ptable[name])
        coords[0].append(float(thisfile[atom].strip().split()[1]))
        coords[1].append(float(thisfile[atom].strip().split()[2]))
        coords[2].append(float(thisfile[atom].strip().split()[3]))
    return atomZ, np.array(coords)


def readRho(rhofile, norbs):
    with open(rhofile, 'rb') as f:
        rho = np.fromfile(f,dtype='complex128',count=norbs*norbs)
        #coords = np.fromfile(f,dtype='float64',count=natoms*3)
        #veloc = np.fromfile(f,dtype='float64',count=natoms*3)    
    rho = np.reshape(rho,(norbs,norbs))
    return rho


def writeCube(cubefile, dens, natoms, box, nx, ny, nz, dx, dy, dz, atomZ, coords):
    atob=1./(constants.physical_constants['atomic unit of length'][0] * 1.0e10)
    with open(cubefile, 'w') as cubout:
        cubout.write('Density generated from density matrix calculated by DFTB+ \n')
        cubout.write('Frame = 0 (Ground state) \n')
        cubout.write('{}   {:.3f}  {:.3f}  {:.3f}\n'.format(natoms, box[0][0]*atob, 
                                                          box[1][0]*atob, box[2][0]*atob))
        cubout.write('{}   {:.3f}  {:.3f}  {:.3f}\n'.format(nx, dx*atob, 0., 0.))
        cubout.write('{}   {:.3f}  {:.3f}  {:.3f}\n'.format(ny, 0., dy*atob, 0.))
        cubout.write('{}   {:.3f}  {:.3f}  {:.3f}\n'.format(nz, 0., 0., dz*atob))
        for iat in range(natoms):
            cubout.write('{}   {:.3f}  {:.3f}  {:.3f}  {:.3f}\n'.format(
                atomZ[iat], 0., coords[0,iat]*atob, coords[1,iat]*atob, coords[2,iat]*atob))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cubout.write('{:.5f} \t'.format(dens[ix, iy, iz]))
                    if iz % 6 == 5:
                        cubout.write('\n')
                cubout.write('\n')


class atomBasis():
    def __init__(self, nelem):
        self.lmax, self.occ, self.cutoff, self.nexp, self.exps, self.ncoeff, \
            self.coeffs = readsto.readStoDataNew(wfc_filename)
        # if no hsd-parser is installed and want to use the old STO parameter
        # files, comment previous two lines and uncomment the next ones:
        # self.lmax, self.occ, self.cutoff, self.nexp, self.exps, self.ncoeff, \
        #    self.coeffs = readsto.readStoDataOld(sto_filename, nelem)


def getBox(coords):
    bspace = 3.5
    box = []
    box.append([min(coords[0,:])-bspace, max(coords[0,:])+bspace])
    box.append([min(coords[1,:])-bspace, max(coords[1,:])+bspace])
    box.append([min(coords[2,:])-bspace, max(coords[2,:])+bspace])
    return box # box = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]


@jit(nopython=True)
def sto(z, l, r, nexp, ncoeff, exps, coeffs):
    sto = 0.0
    for i in range(nexp[l]):
        for j in range(ncoeff[l]):
            sto += coeffs[l][i][j] * r**(l+j-1) * np.exp(-exps[l][i] * r)
    return sto


@jit(nopython=True)
def Rty(n, m, coord, rr):
    xx = coord[0]
    yy = coord[1]
    zz = coord[2]
    if n == 0:
        rty = 0.2820947917738782
    if n == 1:
        if m == -1:
            rty = 0.4886025119029198 * yy / rr
        if m == 0:
            rty = 0.4886025119029198 * zz / rr
        if m == 1:
            rty = 0.4886025119029198 * xx / rr
    if n == 2:
        if m == -2:
            rty = 1.092548430592079 * xx * yy / rr**2
        if m == -1:
            rty = 1.092548430592079 * yy * zz / rr**2
        if m == 0:
            rty = -0.3153915652525200 * (-2.0 * zz**2 + xx**2 + yy**2) / rr**2
        if m == 1:
            rty = 1.092548430592079 * xx * zz / rr**2
        if m == 2:
            rty = 0.5462742152960395 * (xx**2 - yy**2) / rr**2
    if n == 3:
        if m == -3:
            rty = 0.5900435899266435 * yy * (3.0 * xx**2 - yy**2) / rr**3
        if m == -2:
            rty = 2.890611442640554 * xx * yy *zz / rr**3
        if m == -1:
            rty = -0.4570457994644658 * (-4.0 * zz**2 + xx**2 + yy**2) * yy / rr**3
        if m == 0:
            rty = -0.3731763325901155 * zz *(-2.0 * zz**2 + 3.0 * xx**2 + 3.0 * yy**2)/ rr**3
        if m == 1:
            rty = -0.4570457994644658 * (-4.0 * zz**2 + xx**2 + yy**2) * xx / rr**3
        if m == 2:
            rty = 1.445305721320277 * zz * (xx**2 - yy**2) / rr**3
        if m == 3:
            rty = 0.5900435899266435 * xx * (xx**2 - 3.0 * yy**2) / rr**3
    return rty


def fillVectorAtR(atomzs, coords, rx, ry, rz, lmax, cutoff, nexp, ncoeff, exps, coeffs, norbs, orbidx):
    vec = np.zeros(norbs)
    rrorig = np.array([rx,ry,rz])
    rr = coords - rrorig[:,None]
    rrmod = np.linalg.norm(rr, axis=0)
    for iat, atz in enumerate(atomzs):
        if rrmod[iat] < cutoff[atz][0]: #taking the l=0 since cutoff is the same for all l
            idx = 0
            for ll in range(lmax[atz] + 1):
                for mm in range(-ll, ll + 1):
                    eval_wfc = sto(atz, ll, rrmod[iat], nexp[atz], ncoeff[atz], exps[atz], coeffs[atz])
                    eval_wfc = eval_wfc * Rty(ll, mm, -rr[:,iat], rrmod[iat])
                    vec[orbidx[iat]+idx] = eval_wfc
                idx += 1
    return vec


def calculateDensity(rho, atomsz, coords, xx, yy, zz, lmax, cutoff, nexp, ncoeff, exps, coeffs, norbs, orbidx):
    vec1 = fillVectorAtR(atomsz, coords, xx, yy, zz, lmax, cutoff, nexp, ncoeff, exps, coeffs, norbs, orbidx)
    dens = np.dot(np.dot(rho, vec1), vec1)
    return dens.real


def getDensOnGrid(basis, box, nx, ny, nz, dx, dy, dz, rho, atomzs, coords, norbs, orbidx):
    dens = np.zeros((nx, ny, nz))
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                xx = box[0][0] + ix * dx
                yy = box[1][0] + iy * dy
                zz = box[2][0] + iz * dz
                dens[ix, iy, iz] = calculateDensity(rho, atomzs, coords, xx, yy, zz, basis.lmax, \
                                                    basis.cutoff, basis.nexp, basis.ncoeff, \
                                                    basis.exps, basis.coeffs, norbs, orbidx)
    return dens

def binToCube(ifr):
    rhofile = DUMPBIN_DIR+'{}ppdump.bin'.format(ifr)
    rhot = readRho(rhofile, norbs)
    denst = getDensOnGrid(basis, box, nx, ny, nz, dx, dy, dz, rhot, atomZ, myCoords, norbs, orbidx)
    # writes DENSITY DIFFERENCE to cubefile
    writeCube(CUBES_DIR+'{}dens.cube'.format(ifr), denst-inidens, natoms, box, nx, ny, nz, dx, dy, dz, atomZ, myCoords)
    return ifr

##### START CALCULATION #######


basis = atomBasis(nelemread)
atomZ, myCoords = readCoords(coordfile)
box = getBox(myCoords)
dx = (box[0][1]-box[0][0])/float(nx)
dy = (box[1][1]-box[1][0])/float(ny)
dz = (box[2][1]-box[2][0])/float(nz)
norbs = getNorbs(atomZ, basis.lmax)
natoms = len(atomZ)
orbidx = {}
idx = 0
for i in range(natoms):
    orbidx[i] = idx
    idx += sum(orbs_per_l[:basis.lmax[atomZ[i]]+1]) 


rho0 = readRho(DUMPBIN_DIR+rhodumpfile, norbs)
inidens = getDensOnGrid(basis, box, nx, ny, nz, dx, dy, dz, rho0, atomZ, myCoords, norbs, orbidx)

writeCube(CUBES_DIR+'inidens.cube', inidens, natoms, box, nx, ny, nz, dx, dy, dz, atomZ, myCoords)
print('Done frame')

listofframes = list(range(iniframe, endframe+1, frameinterval))

# with concurrent.futures.ProcessPoolExecutor(1) as executor:
#     for it, frameout in enumerate(executor.map(binToCube, listofframes)):
#         print('Done frame',frameout)

for it, frameout in enumerate(listofframes):
    print('Starting', frameout)
    binToCube(it)
    print('Done frame', frameout)


print('Done.')
