import numpy as np
import scipy
from scipy.sparse import linalg, issparse
import subprocess
from os.path import dirname, join as pjoin
import scipy.io as sio

def SolveMatrix(sparseK, b, solver):
    if solver == 'MATLAB':
        sio.savemat('MatrixSolving.mat',{'sparseK':sparseK, 'b':b})
        
        flag = 1
        iter = 1
        while flag and iter<10:
            try:
                print('Running MATLAB...')
                flag = subprocess.call("commandsolving.bat")
                iter += 1
            except:
                return False
        if iter == 10:
            print('Too many trials...')
            return False
        
        read_data = sio.loadmat('phisolved.mat')
        phi = read_data['phi']
        #phi = phi.todense()
        if issparse(phi) is True:
            phi = phi.todense()

        if phi.shape[1] == 1:
            phi = phi.flatten()
    else:
        phi = linalg.spsolve(sparseK.tocsc(), b)
    return phi
