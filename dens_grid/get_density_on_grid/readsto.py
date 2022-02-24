"""readsto.py: Parses STO parameters from different file formats: plain file
 format or HSD format. For HSD files (like wfc*.hsd) files for waveplots,
needs hsd-python module (https://github.com/dftbplus/hsd-python)  """

__author__      = "Franco Bonafe"
__email__       = "fbonafe@unc.edu.ar"
__copyright__   = "Copyright 2018, FCQ UNC"
__status__      = "Prototype"

import itertools
from collections import defaultdict
import numpy as np

def readStoDataNew(wfc_filename):
    import hsd
    lmax = {}
    occ = defaultdict(list)
    cutoff = defaultdict(list)
    nexp = defaultdict(list)
    exps = defaultdict(list)
    ncoeff = defaultdict(list)
    coeffs = defaultdict(list)

    dictbuilder = hsd.HsdDictBuilder()
    parser = hsd.HsdParser(eventhandler=dictbuilder)
    with open(wfc_filename, 'r') as fileobj:
        parser.parse(fileobj)
    pyrep = dictbuilder.hsddict
    for key, val in pyrep.items():
        # print(key)
        atz = int(pyrep[key]['AtomicNumber'])
        if isinstance(pyrep[key]['Orbital'], dict):
            lmax[atz] = 0
            pyrep[key]['Orbital'] = [pyrep[key]['Orbital']]
        else:
            lmax[atz] = (pyrep[key]['Orbital'][-1]['AngularMomentum'])

        for iorb, orbs in enumerate(pyrep[key]['Orbital']):
            ll = orbs['AngularMomentum']
            occ[atz].append(orbs['Occupation'])
            cutoff[atz].append(orbs['Cutoff'])
            if any(isinstance(exp, list) for exp in orbs['Exponents']):
                exps[atz].append(list(itertools.chain(*orbs['Exponents'])))
            else:
                exps[atz].append(list(itertools.chain(orbs['Exponents'])))
            number_of_exp = len(exps[atz][iorb])
            nexp[atz].append(number_of_exp)
            flat_coeffs = list(itertools.chain(*orbs['Coefficients']))
            total_coeffs = len(flat_coeffs)
            ncoeffs_per_exp = total_coeffs // number_of_exp
            coeffs[atz].append([flat_coeffs[ic:ic+ncoeffs_per_exp] \
                            for ic in range(0, len(flat_coeffs), ncoeffs_per_exp)])
            ncoeff[atz].append(ncoeffs_per_exp)

        occ[atz] = np.array(occ[atz])
        cutoff[atz] = np.array(cutoff[atz])
        nexp[atz] = np.array(nexp[atz])
        exps[atz] = np.array(exps[atz])
        ncoeff[atz] = np.array(ncoeff[atz])
        coeffs[atz] = np.array(coeffs[atz])
    return lmax, occ, cutoff, nexp, exps, ncoeff, coeffs
            
def readStoDataOld(sto_filename, nelemread):
    lmax = {}
    occ = defaultdict(list)
    cutoff = defaultdict(list)
    nexp = defaultdict(list)
    exps = defaultdict(list)
    ncoeff = defaultdict(list)
    coeffs = defaultdict(list)
    with open(sto_filename, 'r') as stofile:  # file with the STO parameters
        for ielem in range(nelemread):         # based on the waveplot wfc files
            atz = int(stofile.readline().split()[0])
            lmax[atz] = int(stofile.readline().split()[0])

            for l in range(lmax[atz]+1):
                ll = int(stofile.readline().split()[0])
                occ[atz].append(float(stofile.readline().split()[0]))
                cutoff[atz].append(float(stofile.readline().split()[0]))
                this_nexp = int(stofile.readline().split()[0])
                nexp[atz].append(this_nexp)

                exps_line = stofile.readline().split()
                exps[atz].append([float(exps_line[j]) for j in range(this_nexp)])
                this_ncoeff = int(stofile.readline().split()[0])
                ncoeff[atz].append(this_ncoeff)
                coeffs_line = stofile.readline().split()
                lcoeffs_aux = []
                for j in range(this_nexp):
                    lcoeffs_aux.append([float(coeffs_line[j*3+k]) \
                                        for k in range(this_ncoeff)])
                coeffs[atz].append(lcoeffs_aux)
            
    return lmax, occ, cutoff, nexp, exps, ncoeff, coeffs

def test():
    lmax, occ, cutoff, nexp, exps, ncoeff, \
        coeffs = readStoDataNew('wfc.mio-1-1.hsd')
    lmax2, occ2, cutoff2, nexp2, exps2, ncoeff2, \
        coeffs2 = readStoDataOld('STO.mio.dat', 5)
    print('are they equal?')
    for key in lmax:
        print('key',key)
        print('lmax')
        print(lmax[key] == lmax2[key])
        print('occ')
        print(occ[key] == occ2[key])
        print('cutoff')
        print(cutoff[key] == cutoff2[key])
        print('nexp')
        print(nexp[key] == nexp2[key])
        print('exps')
        print(exps[key] == exps2[key])
        print('ncoeff')
        print(ncoeff[key] == ncoeff2[key])
        print(ncoeff[key], ncoeff2[key])
        print('coeffs')
        print(coeffs[key] == coeffs2[key])
        print('\n')    

if __name__ == '__main__':
    test()
