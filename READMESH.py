# READMESH.py, 100823

import numpy as np
class MESH:
    def __init__(self, filepath):
        
        self.filepath = filepath
        firstrow = np.loadtxt(filepath, max_rows = 1, dtype = int)
        self.nv = firstrow[0]
        self.nt = firstrow[1]
        self.ns = firstrow[2]
        self.vertices = np.loadtxt(filepath, skiprows=1, max_rows = self.nv, dtype = np.float64)
        self.elements = np.loadtxt(filepath, skiprows=1+self.nv, max_rows = self.nt, dtype = int)
        self.boundary = np.loadtxt(filepath, skiprows=1+self.nv+self.nt, max_rows = self.ns, dtype = int)