##!/usr/bin python3

"""density_on_grid_timeprop_dftb.py: Calculates electronic density difference in real space grid \
   and saves it as CUBE files, from density matrix dumped in time propagations using DFTB+.
   Needs wfc file for the SK parameter set, downloadable from dftb.org."""

__authors__      = "Franco Bonafe, Cristian G. Sanchez"
__maintainers__  = "Franco Bonafe, Carlos Medrano"
__email__       = "fbonafe@unc.edu.ar, cmedrano@unc.edu.ar"
__copyright__   = "Copyright 2018, FCQ UNC"
__status__      = "Prototype"

import os
from collections import defaultdict
import numpy as np

from numba import jit   ###njit
# from numba.typed import List
from scipy.special import sph_harm as Ynm #Ynm(m, n, theta, phi), theta=azimuthal, phi=polar
from scipy import constants
import readsto
import concurrent.futures
from ase.io import read

###### Variables to set before running (EDIT) ############# 


#grid points
nx = 48
ny = 12
nz = 12

iniframe = 5          #initial and final frames (depends on the dftb input)
endframe = 5
frameinterval = 1     #interval of frames to take into account


DUMPBIN_DIR = '../pump_frames/'  # directory where the *dump.bin files are located
#CUBES_DIR = '.' # directory where the cubefiles will be stored
CUBES_DIR = './cubes/' # directory where the cubefiles will be stored

coordfile = '../geom.in.gen'  ## edit accordingly
rhodumpfile = '0ppdump.bin' # *dump.bin file for step = 0 (ground state density), edit

wfc_filename = 'wfc.3ob-3-1.hsd'
au__to__fs = 1/0.413413733365614E+02      ## from manual
ang__to__bohr = 1./(constants.physical_constants['atomic unit of length'][0] * 1.0e10)
##########################################################


### Definition of functions #######################

nelemread = 1             #number of elements in the system

orbs_per_l = [1, 3, 5, 7]    #l quantum number for each type of orbital (do not change)

if not os.path.exists(CUBES_DIR):
    os.makedirs(CUBES_DIR)

def getNorbs(atomzs, lmax):
    """
    This function return the total number of orbitals in the system based on the
    atomic numbers and the maxAngMomentum of the specific wfc file (based on a
    specific SKF set of parameters) 
    """
    norbs = 0
    for iat in atomzs:
        for ll in range(lmax[iat]+1):
            norbs += orbs_per_l[ll]
    return norbs

def readCoords(thisfile):     # This function use ASE
    """
    This function read the coord.gen file to get and return the atomic positions
    and the atomic numbers.
    """
    mol = read(thisfile)        
    pos = mol.get_positions() # [iat, idir]
    coords = np.transpose(pos)
    atomZ = mol.get_atomic_numbers()
    return atomZ, np.array(coords)

def readRho(rhofile, norbs):
    """
    This function read the ppdump.bin file (output from DFTB+) and return
    the corresponding density matrix <rho> in square format [norbs,norbs] 
    """
    try:
        with open(rhofile, 'rb') as f:
            dumpfmt = np.fromfile(f,dtype='int32',count=1)
            norbsfromrho = np.fromfile(f,dtype='int32',count=1)
            nspin = np.fromfile(f,dtype='int32',count=1)
            natoms = np.fromfile(f,dtype='int32',count=1)
            time = np.fromfile(f,dtype='float64',count=1)
            dt = np.fromfile(f, dtype='float64', count=1)
            print('dumpfmt:',dumpfmt)
            print('norbs:',norbsfromrho)
            print('nspin:',nspin)
            print('natoms:',natoms)
            print('time in fs:',time*au__to__fs)
            print('dt in au:',dt)
            rho = np.fromfile(f,dtype='complex128',count=norbs*norbs)
    except IOError:
        print('File',rhofile,'not exist')
    rho = np.reshape(rho,(norbs,norbs))
    return rho


def writeCube(cubefile, dens, natoms, box, nx, ny, nz, dx, dy, dz, atomZ, coords, ang__to__bohr):
#    ang__to__bohr = 1./(constants.physical_constants['atomic unit of length'][0] * 1.0e10)
    with open(cubefile, 'w') as cubout:
        cubout.write('Density generated from density matrix calculated by DFTB+ \n')
        cubout.write('Frame = 0 (Ground state) \n')
        cubout.write('{}   {:.3f}  {:.3f}  {:.3f}\n'.format(natoms, box[0][0]*ang__to__bohr, 
                                                          box[1][0]*ang__to__bohr, box[2][0]*ang__to__bohr))
        cubout.write('{}   {:.3f}  {:.3f}  {:.3f}\n'.format(nx, dx*ang__to__bohr, 0., 0.))
        cubout.write('{}   {:.3f}  {:.3f}  {:.3f}\n'.format(ny, 0., dy*ang__to__bohr, 0.))
        cubout.write('{}   {:.3f}  {:.3f}  {:.3f}\n'.format(nz, 0., 0., dz*ang__to__bohr))
        for iat in range(natoms):
            cubout.write('{}   {:.3f}  {:.3f}  {:.3f}  {:.3f}\n'.format(
                atomZ[iat], 0., coords[0,iat]*ang__to__bohr, coords[1,iat]*ang__to__bohr, coords[2,iat]*ang__to__bohr))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cubout.write('{:.8f} \t'.format(dens[ix, iy, iz]))
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
    """
    This function construct a box for the calculation of the charge density
    based on the min and max values of the X,Y and Z coordinates of the system.
    """
    bspace = 3.5
    box = []
    box.append([min(coords[0,:])-bspace, max(coords[0,:])+bspace])
    box.append([min(coords[1,:])-bspace, max(coords[1,:])+bspace])
    box.append([min(coords[2,:])-bspace, max(coords[2,:])+bspace])
    return box # box = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]


#@jit(nopython=True)
def sto(z, l, r, nexp, ncoeff, exps, coeffs):
    """
    This function calculate the radial part of the wavefunction 
    in a certain position r
    """
    sto = 0.0
    for i in range(nexp[l]):
        for j in range(ncoeff[l]):
            sto += coeffs[l][i][j] * r**(l+j-1) * np.exp(-exps[l][i] * r) #radial part
    return sto

#@jit(nopython=True)
def Rty(n, m, coord, rr):
    """
    This function calculate the angular part of the wavefunction
    with spherical harmonics in a certain position rr
    """
    xx = coord[0]
    yy = coord[1]
    zz = coord[2]
    #Tabulated spherical harmonics (find reference)
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
    rr = rrorig[:,None] - coords
    rrmod = np.linalg.norm(rr, axis=0)
    for iat, atz in enumerate(atomzs):
        if rrmod[iat] < cutoff[atz][0]: #taking the l=0 since cutoff is the same for all l
            idx = 0
            for ll in range(lmax[atz] + 1):
                for mm in range(-ll, ll + 1):
                    if rrmod[iat] > 1e-1:
                        eval_wfc_sto = sto(atz, ll, rrmod[iat], nexp[atz], ncoeff[atz], exps[atz], coeffs[atz])
                    else:
                        eval_wfc_sto = 0.0        #To avoid divergence near the nucleus
                    eval_wfc = eval_wfc_sto * Rty(ll, mm, rr[:,iat], rrmod[iat])
#                    if eval_wfc > 3 or eval_wfc < -3:
#                        print(" ")
#                        print("************* TESTING VECTOR1 ***********")
#                        print("eval_wfc: ", eval_wfc)
#                        print(" ")
#                        print("eval_wfc_sto: ", eval_wfc_sto)
#                        print(" ")
#                        print("Position: ", rr)
#                        print(" ")
#                        print("Modulus distance: ", rrmod[iat])
#                       print(" ")
#                       print("Origin position: ", rrorig)
#                       print(" ")
                    vec[orbidx[iat]+idx] = eval_wfc
                idx += 1
    return vec


def calculateDensity(rho, atomsz, coords, xx, yy, zz, lmax, cutoff, nexp, ncoeff, exps, coeffs, norbs, orbidx):
    vec1 = fillVectorAtR(atomsz, coords, xx, yy, zz, lmax, cutoff, nexp, ncoeff, exps, coeffs, norbs, orbidx)
    dens = np.dot(np.dot(rho, vec1), vec1)
    if dens.real > 200:
        print(" ")
        print("************* TESTING DENSITY ***********")
        print("density value:", dens.real)
        print("max value of rho:", np.max(rho))
        print("min value of rho:", np.min(rho))
        print("vector 1")
        print(vec1)
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
    print(rhofile)
    rhot = readRho(rhofile, norbs)
    denst = getDensOnGrid(basis, box, nx, ny, nz, dx, dy, dz, rhot, atomZ, myCoords, norbs, orbidx)
    # writes DENSITY DIFFERENCE to cubefile
    writeCube(CUBES_DIR+'{}dens.cube'.format(ifr), denst-inidens, natoms, box, nx, ny, nz, dx, dy, dz, atomZ, myCoords, ang__to__bohr)
    return ifr

##### START CALCULATION #######


basis = atomBasis(nelemread)              #use readsto to get the basis
atomZ, myCoords = readCoords(coordfile)   #get the atomic numbers and positions
box = getBox(myCoords)                    #create the box for the density calculation
dx = (box[0][1]-box[0][0])/float(nx)      #deltas in x, y, z based on the box dimension
dy = (box[1][1]-box[1][0])/float(ny)      #and the grid points
dz = (box[2][1]-box[2][0])/float(nz)
dV = dx*(ang__to__bohr)*dy*(ang__to__bohr)*dz*(ang__to__bohr) #delta Volume in bohr^3
print("dx ", dx)
print("dy ", dy)
print("dz ", dz)
norbs = getNorbs(atomZ, basis.lmax)       #basis.lmax is a dictionary {Z:maxAngMomentum}   
natoms = len(atomZ)
orbidx = {}
idx = 0
for i in range(natoms):                   #creates a dictionary with {atom idx : 1 orbital idx}  
    orbidx[i] = idx
    idx += sum(orbs_per_l[:basis.lmax[atomZ[i]]+1])

rho0 = readRho(DUMPBIN_DIR+rhodumpfile, norbs)   #read initial density matrix rho0
inidens = getDensOnGrid(basis, box, nx, ny, nz, dx, dy, dz, rho0, atomZ, myCoords, norbs, orbidx)

writeCube(CUBES_DIR+'inidens.cube', inidens, natoms, box, nx, ny, nz, dx, dy, dz, atomZ, myCoords, ang__to__bohr)
print('Done initial density')
print('Number of electrons:', np.sum(inidens)*dV)
print('i Volume:', dV)


listofframes = list(range(iniframe, endframe+1, frameinterval))
print('list of frames',listofframes)
# with concurrent.futures.ProcessPoolExecutor(1) as executor:
#     for it, frameout in enumerate(executor.map(binToCube, listofframes)):
#         print('Done frame',frameout)

for frame in listofframes:
    print('Starting', frame)
    binToCube(frame)
    print('Done frame', frame)


print('Done.')
