# FEM2D.py, 102623

import numpy as np
import torch
import time
from torch import autograd
from scipy import sparse
#from scipy.sparse import linalg
from FEM_Kernel import *
from solvematrix import SolveMatrix
PI = np.pi

class FEM2D:
    def __init__(self, mesh, vh, ord, device, solver):
        self.ord = ord
        self.device = device
        self.mesh = mesh
        self.vh = vh
        self.ndof= np.max(vh) + 1
        if self.ord == 'P1':
            self.dofe = 3
            self.dofedge = 2
        elif self.ord == 'P2':
            self.dofe = 6
            self.dofedge = 3
        
        c0 = 3e8
        f0 = 1e9    #
        self.epi = 4.   #
        self.wavelength0 = c0/f0
        self.k0 = 2*PI*f0/c0
        self.epi0 = 1.
        self.epi_label = 0  #
        
        self.E0 = 1.
        self.labelleft = 1  #
        self.labeltop = 2
        self.labelright = 3 #
        self.labelbottom = 4
        self.qleft = 0
        self.qright = 0
        #self.gamma1 = 0
        #self.gamma2 = 0
        self.gamma1 = 1.00023*1j*self.k0
        self.gamma2 = 0.51555*1j/self.k0
        #self.gamma = 0
        
        
        self.epi_list = torch.ones(self.mesh.nt, dtype = torch.complex128, device = self.device)
        self.index_inout = self.mesh.elements[:,-1].flatten() == self.epi_label
        self.epi_list[self.index_inout] = self.epi
        self.alphax = torch.ones_like(self.epi_list, requires_grad = False, device = self.device)
        self.alphay = torch.ones_like(self.epi_list, requires_grad = False, device = self.device)
        self.beta0 = -self.k0**2
        self.beta = self.beta0 * self.epi_list
        
        self.solver = solver
        #self.FE = torch.zeros(self.mesh.nt, dtype = torch.complex128, device = self.device)
        self.FEfunc = lambda x: self.k0**2 * Uinc(x, self.k0)
        
    def calc_K(self):
        
        t0 = time.time()
        self.ROWB = np.array([], dtype = int)
        #XI = np.zeros((self.mesh.nt, 3), dtype = float)
        XI = self.mesh.elements[:, :3] - 1
        
        # XTCCOR = [vertices0, vertices1, vertices2], verticesx[nt, 2] = [x_coor[nt], y_coor[nt]]
        vertices = torch.tensor(self.mesh.vertices, dtype = torch.float64, requires_grad = False, device = self.device)
        # XTCOOR = [Xcoor, Ycoor], Xcoor[nt, 3] = [point0[1:nt], point1[1:nt], point2[1:nt]]
        self.XTCOORX = torch.t(torch.stack((vertices[XI[:,0], 0], vertices[XI[:,1], 0], vertices[XI[:,2], 0])) )
        self.XTCOORY = torch.t(torch.stack((vertices[XI[:,0], 1], vertices[XI[:,1], 1], vertices[XI[:,2], 1])) )
        self.XTCOOR = [self.XTCOORX, self.XTCOORY]
        if self.ord == 'P1':
            self.KE, self.BE = K_P1(self.XTCOOR, self.alphax, self.alphay, self.beta, self.epi_list, self.FEfunc, self.device)
        elif self.ord == 'P2':
            self.KE, self.BE = K_P2(self.XTCOOR, self.alphax, self.alphay, self.beta, self.epi_list, self.FEfunc, self.device)
        vh2 = self.vh.reshape(self.mesh.nt, self.dofe)
        VHstack = [vh2 for _ in range(self.dofe)]
        self.ROW = np.stack(VHstack, axis=2).flatten()
        self.COL = np.stack(VHstack, axis=1).flatten()
        self.ROWB = self.vh
        self.KE = self.KE.flatten()
        self.BE = self.BE.flatten()
        
        print('Inner elapsed time is: ', time.time()-t0)
        #self.KE = torch.tensor(self.KE, requires_grad = False)
        #self.BE = torch.tensor(self.BE, requires_grad = False)
        return self.KE
        
    def get_boundary_elements(self, label):
        boundary_index = np.nonzero(self.mesh.boundary[:,-1] == label)[0]
        vertice_index = self.mesh.boundary[boundary_index, 0:2].flatten()
        vertice_index = np.unique(vertice_index)
        element_vertices_mask = np.isin(self.mesh.elements[:, :3], vertice_index)
        element_index = np.nonzero(np.sum(element_vertices_mask.view(dtype=np.int8), axis=1) == 2)[0]
        return vertice_index, element_index
        
    def get_boundary_index(self, label):
        vertice_index, element_index = self.get_boundary_elements(label)
        vh = self.vh.reshape(self.mesh.nt, self.dofe)
        vh_related = vh[element_index, :]
        
        XI_related = self.mesh.elements[element_index, :3] - 1
        nt = XI_related.shape[0]
        
        para_mask = np.nonzero(np.isin(XI_related + 1, vertice_index))
        #print(para_mask.shape)

        index = np.arange(3)*np.ones((nt,3), dtype = int)
        x_index = XI_related[para_mask].reshape(nt, 2)
        pointindex = index[para_mask].reshape(nt, 2)
        if self.ord == 'P1':
            pointindex = (pointindex[:,0], pointindex[:,1])
        elif self.ord == 'P2':
            pointindex = (pointindex[:,0], 6-(pointindex[:,0]+pointindex[:,1]), pointindex[:,1])
        vh_index = np.empty([nt, self.dofedge], dtype = int)
        for n in range(self.dofedge):
            vh_index[:, n] = vh_related[(range(nt), pointindex[n])]
        # x_index = [point1_meshindex, point2_meshindex], vh_index = [point1_vh, midpoint_vh, point2_vh]
        return x_index, vh_index
        
    def calc_boundary(self, label, gamma1, gamma2, q, device):
        x_index, vh_index = self.get_boundary_index(label)
        vertices = torch.tensor(self.mesh.vertices, requires_grad = False, device = device)
        # XTCCOR = [Xcoor, Ycoor], Xcoor[nt, 3] = [point1[1:nt], point2[1:nt], point1[1:nt]]
        XTCOORX = torch.t(torch.stack((vertices[x_index[:,0], 0], vertices[x_index[:,1], 0])) )
        XTCOORY = torch.t(torch.stack((vertices[x_index[:,0], 1], vertices[x_index[:,1], 1])) )
        XTCOOR = [XTCOORX, XTCOORY]
        ls = torch.sqrt((XTCOOR[0][:,1]-XTCOOR[0][:,0])**2+(XTCOOR[1][:,1]-XTCOOR[1][:,0])**2)
        ls = torch.unsqueeze(ls, dim = 1)
        if self.ord == 'P1':
            KSEmat1 = [[2/6, 1/6], [1/6, 2/6]]
            KSEmat2 = [[-1., 1.],[1., -1.]]
            bsmat = [1/2, 1/2]
        elif self.ord == 'P2':
            KSEmat1 = [[2/15, 1/15, -1/30], \
                    [1/15, 8/15, 1/15], \
                    [-1/30, 1/15, 2/15]]
            KSEmat2 = [[7/3, -8/3, 1/3], \
                       [-8/3, 16/3, -8/3], \
                        [1/3, -8/3, 7/3]]
            bsmat = [1/6, 2/3, 1/6]
        KSEmat1 = torch.tensor(KSEmat1, dtype = torch.complex128, requires_grad = False, device = self.device)
        KSEmat2 = torch.tensor(KSEmat2, dtype = torch.complex128, requires_grad = False, device = self.device)
        bsmat = torch.tensor(bsmat, dtype = torch.complex128, requires_grad = False, device = self.device)
        KSE = ls * gamma1 * KSEmat1.flatten() - gamma2/ls*KSEmat2.flatten()
        vhstack = [vh_index for _ in range(self.dofedge)]
        rows = np.stack(vhstack, axis = 2).flatten()
        cols = np.stack(vhstack, axis = 1).flatten()

        BSE = ls * q * bsmat.flatten()
        rowbs = vh_index.flatten()
        return rows, cols, rowbs, KSE.flatten(), BSE.flatten()
            

    def calc_B(self):
        rows, cols, rowbs, KSE, BSE = self.calc_boundary(self.labelleft, self.gamma1, self.gamma2, self.qleft, self.device)
        self.rowKs = rows
        self.colKs = cols
        self.rowbs = rowbs
        self.Kse = KSE
        self.bse = BSE
        
        rows, cols, rowbs, KSE, BSE = self.calc_boundary(self.labelright, self.gamma1, self.gamma2, self.qright, self.device)
        self.rowKs = np.concatenate((self.rowKs, rows), axis = None)
        self.colKs = np.concatenate((self.colKs, cols), axis = None)
        self.rowbs = np.concatenate((self.rowbs, rowbs), axis = None)
        self.Kse = torch.cat((self.Kse, KSE))
        self.bse = torch.cat((self.bse, BSE))

        rows, cols, rowbs, KSE, BSE = self.calc_boundary(self.labeltop, self.gamma1, self.gamma2, self.qright, self.device)
        self.rowKs = np.concatenate((self.rowKs, rows), axis = None)
        self.colKs = np.concatenate((self.colKs, cols), axis = None)
        self.rowbs = np.concatenate((self.rowbs, rowbs), axis = None)
        self.Kse = torch.cat((self.Kse, KSE))
        self.bse = torch.cat((self.bse, BSE))

        rows, cols, rowbs, KSE, BSE = self.calc_boundary(self.labelbottom, self.gamma1, self.gamma2, self.qright, self.device)
        self.rowKs = np.concatenate((self.rowKs, rows), axis = None)
        self.colKs = np.concatenate((self.colKs, cols), axis = None)
        self.rowbs = np.concatenate((self.rowbs, rowbs), axis = None)
        self.Kse = torch.cat((self.Kse, KSE))
        self.bse = torch.cat((self.bse, BSE))
        
        return self.Kse
    
    def solve_system(self, Kinput = None, binput = None):
        if Kinput is None:
            rowtotal = np.concatenate((self.ROW, self.rowKs), axis = None)
            coltotal = np.concatenate((self.COL, self.colKs), axis = None)
            Ktotal = torch.cat((self.KE, self.Kse))
            Ktotal = Ktotal.cpu().detach().numpy()
        else:
            rowtotal = Kinput[0].astype(int)
            coltotal = Kinput[1].astype(int)
            Ktotal = Kinput[2]
        self.sparseK = sparse.coo_matrix((Ktotal, (rowtotal, coltotal)), shape = (self.ndof, self.ndof))
        if binput is None:
            rowbtotal = np.concatenate((self.ROWB, self.rowbs), axis = None)
            colbtotal = np.zeros_like(rowbtotal)
            btotal = torch.cat((self.BE, self.bse))
            btotal = btotal.cpu().detach().numpy()
            self.sparseb = sparse.coo_matrix((btotal, (rowbtotal, colbtotal)), shape = (self.ndof, 1))
            #self.sparseb = self.sparseb.tocsc()
            self.phi = SolveMatrix(self.sparseK, self.sparseb, solver = self.solver)
        else:
            btotal = binput
            self.phi = SolveMatrix(self.sparseK, btotal, solver=self.solver)
        return self.phi
    
    def eval_phix(self, x, phi):
        # XTCOOR = [Xcoor, Ycoor], Xcoor[nt, 3] = [point0[1:nt], point1[1:nt], point2[1:nt]]
        elementindex = isInside(x, self.XTCOOR)
        vh_element = self.vh.reshape(self.mesh.nt, self.dofe)
        vh_element = vh_element[elementindex]
        phi_element = phi[vh_element]
        XTCOORXx = self.XTCOOR[0][elementindex,:].view(1,3)
        XTCOORYx = self.XTCOOR[1][elementindex,:].view(1,3)
        XTCOORx = [XTCOORXx, XTCOORYx]
        AEx, BEx, CEx = interp2d(XTCOORx)
        AREAx = (BEx[:, 0] * CEx[:, 1] - BEx[:, 1] * CEx[:, 0]) / 2
        Lx = (AEx+BEx*x[0]+CEx*x[1])/(2*AREAx)
        Lx = Lx.flatten()
        index = [[1,2],[2,0],[0,1]]
        phix = 0.+0.j
        for k in range(3):
            phix += phi_element[k]*(2*Lx[k]-1)*Lx[k]
            phix += phi_element[k+3]*(4*Lx[index[k][0]]*Lx[index[k][1]])
        
        return phix
        
    def calc_R(self, phi):

        x_index, vh_index = self.get_boundary_index(self.labelleft)
        nt = vh_index.shape[0]
        
        self.mode_pattern = np.ones(nt, dtype = complex)
        
        self.boundaryL = 0
        self.boundaryphi = np.array([], dtype = complex)
        
        vertices = self.mesh.vertices
        XTCOORX = np.vstack((vertices[x_index[:,0], 0], vertices[x_index[:,1], 0])).transpose()
        XTCOORY = np.vstack((vertices[x_index[:,0], 1], vertices[x_index[:,1], 1])).transpose()
        XTCOOR = [XTCOORX, XTCOORY]
        ls = np.sqrt((XTCOOR[0][:,0]-XTCOOR[0][:,0])**2+(XTCOOR[1][:,1]-XTCOOR[1][:,0])**2)
        ls = np.expand_dims(ls, axis = 1)
        phi_related = phi[vh_index.flatten()].reshape(nt,self.dofedge)
        if self.ord == 'P1':
            intmat = np.array([1/2, 1/2], dtype = np.float64)
        elif self.ord == 'P2':
            intmat = np.array([1/6, 2/3, 1/6], dtype = np.float64)
        intphi = phi_related * intmat * ls
        intphimode = np.sum(intphi, axis = 1) * self.mode_pattern
        self.boundaryL = np.sum(ls)
        mode_match = np.sum(intphimode)/self.boundaryL
        self.R = (mode_match - self.E0) / self.E0
        self.R2 = self.R * self.R.conj()
        self.R2 = self.R2.real.astype(float)
        return self.R, self.R2
    
class grad:
    def __init__(self, fem2d, solver):
        self.fem = fem2d
        self.device = fem2d.device
        self.ord = fem2d.ord
        self.dofe = fem2d.dofe
        self.vertice_related = []
        self.element_related = []
        
        self.solver = solver
    def get_related_elements(self, label, case):
        self.case = case
        if case == 'geox' or case == 'geoy' or case == 'geor':
            vertice_index, element_index = self.fem.get_boundary_elements(label)
            element_vertices_mask = np.isin(self.fem.mesh.elements[:, :3], vertice_index)
            element_vertice_index = np.nonzero(element_vertices_mask)[0]
            self.element_related_index = np.unique(element_vertice_index)

        elif case == 'eps' :
            self.element_related_index = np.nonzero(self.fem.mesh.elements[:, -1] == label)[0]
        
        XI_related = self.fem.mesh.elements[self.element_related_index, :3] - 1
        self.nt = XI_related.shape[0]
        
        vertices = torch.tensor(self.fem.mesh.vertices, requires_grad = False, device = self.device)
        # XTCCOR = [Xcoor, Ycoor], Xcoor[nt, 3] = [point1[1:nt], point2[1:nt], point1[1:nt]]
        self.XTCOORX = torch.t(torch.stack((vertices[XI_related[:,0], 0], vertices[XI_related[:,1], 0], vertices[XI_related[:,2], 0])) )
        self.XTCOORY = torch.t(torch.stack((vertices[XI_related[:,0], 1], vertices[XI_related[:,1], 1], vertices[XI_related[:,2], 1])) )
        self.XTCOOR = [self.XTCOORX, self.XTCOORY]
        self.alphax = self.fem.alphax[self.element_related_index]
        self.alphay = self.fem.alphay[self.element_related_index]
        self.beta = self.fem.beta[self.element_related_index]
        self.epi_list = self.fem.epi_list[self.element_related_index]
        self.XI = XI_related
        
        vh2 = self.fem.vh.reshape(self.fem.mesh.nt, self.dofe)
        self.vh_related = vh2[self.element_related_index,:]
        
        if case == 'geox' or case == 'geoy' or case == 'geor':
            self.para_mask = np.nonzero(np.isin(XI_related + 1, vertice_index) )
        self.FEfunc = self.fem.FEfunc
            
            
    
    def append_index(self):
        self.ROWB = self.vh_related.flatten()
        VHstack = [self.vh_related for _ in range(self.dofe)]
        self.ROW = np.stack(VHstack, axis=2).flatten()
        self.COL = np.stack(VHstack, axis=1).flatten()
        
    def calc_binc(self, para):
        if self.case == 'geox':
            self.XTCOORX[self.para_mask] += para
        elif self.case == 'geoy':
            self.XTCOORY[self.para_mask] += para
        elif self.case == 'geor':
            theta = torch.atan2(self.XTCOORY, self.XTCOORX)
            self.XTCOORX[self.para_mask] += para*torch.cos(theta[self.para_mask])
            self.XTCOORY[self.para_mask] += para*torch.sin(theta[self.para_mask])
        elif self.case == 'eps':
            self.epi_list += para
            self.beta = self.fem.beta0 * self.epi_list
        self.XTCOOR = [self.XTCOORX, self.XTCOORY]
        if self.ord == 'P1':
            self.KE, self.BE = K_P1(self.XTCOOR, self.alphax, self.alphay, self.beta, self.epi_list, self.fem.FEfunc, self.device)
        elif self.ord == 'P2':
            self.KE, self.BE = K_P2(self.XTCOOR, self.alphax, self.alphay, self.beta, self.epi_list, self.fem.FEfunc, self.device)
        self.KE = self.KE.flatten()
        self.BE = self.BE.flatten()
        
        #self.KE = torch.tensor(self.KE, requires_grad = False)
        #self.BE = torch.tensor(self.BE, requires_grad = False)
        return torch.cat((self.BE.real, self.BE.imag))
        
    
    def calc_K(self, para):
        if self.case == 'geox':
            self.XTCOORX[self.para_mask] += para
        elif self.case == 'geoy':
            self.XTCOORY[self.para_mask] += para
        elif self.case == 'geor':
            theta = torch.atan2(self.XTCOORY, self.XTCOORX)
            self.XTCOORX[self.para_mask] += para*torch.cos(theta[self.para_mask])
            self.XTCOORY[self.para_mask] += para*torch.sin(theta[self.para_mask])
        elif self.case == 'eps':
            self.epi_list += para
            self.beta = self.fem.beta0 * self.epi_list
        self.XTCOOR = [self.XTCOORX, self.XTCOORY]
        if self.ord == 'P1':
            self.KE, self.BE = K_P1(self.XTCOOR, self.alphax, self.alphay, self.beta, self.epi_list, self.fem.FEfunc, self.device)
        elif self.ord == 'P2':
            self.KE, self.BE = K_P2(self.XTCOOR, self.alphax, self.alphay, self.beta, self.epi_list, self.fem.FEfunc, self.device)
        self.KE = self.KE.flatten()
        self.BE = self.BE.flatten()
        
        #self.KE = torch.tensor(self.KE, requires_grad = False)
        #self.BE = torch.tensor(self.BE, requires_grad = False)
        return torch.cat((self.KE.real, self.KE.imag))
        
    def grad_jac(self, para, case, label, external_grad = None):
        self.dKlist = []
        self.dBlist = []
        self.Npara = len(para)
        for k in range(self.Npara):
            self.get_related_elements(label = label[k], case = case[k])
            dK = autograd.functional.jacobian(self.calc_K, para[k], vectorize = True, strategy = 'forward-mode')
            lengthdK = dK.shape[0]
            dK = dK[:int(lengthdK/2)] + 1j*dK[int(lengthdK/2):]
            
            dB = autograd.functional.jacobian(self.calc_binc, para[k], vectorize = True, strategy = 'forward-mode')
            lengthdB = dB.shape[0]
            dB = dB[:int(lengthdB/2)] + 1j*dB[int(lengthdB/2):]
            
            if external_grad is not None:
                dK = torch.matmul(self.dK, external_grad)   # dK = [[dK1/dp1, dK1/dp2], [dK2/dp1, dK2/dp2], ...]
                db = torch.matmul(self.db, external_grad)
            self.append_index()
            dK = dK.cpu().detach().numpy()
            dB = dB.cpu().detach().numpy()
            sparsedK = sparse.coo_matrix((dK, (self.ROW, self.COL)), shape = (self.fem.ndof, self.fem.ndof) )
            colB = np.zeros_like(self.ROWB)
            sparsedB = sparse.coo_matrix((dB, (self.ROWB, colB)), shape = (self.fem.ndof, 1) )
            self.dKlist.append(sparsedK)
            self.dBlist.append(sparsedB)
            
    def grad_phi(self):
        self.dphilist = []
        bright = np.empty([self.fem.ndof, self.Npara], dtype = complex)
        for k in range(self.Npara):
            sparsedK, sparsedB = self.dKlist[k], self.dBlist[k]
            #sparsedK, sparsedB = sparsedK.tocsc(), sparsedB.tocsc()
            temp = sparsedB - np.expand_dims(sparsedK @ self.fem.phi, axis = 1)
            bright[:,k] = temp.flatten()
        dphi = SolveMatrix(self.fem.sparseK, bright, solver = self.solver)
        for k in range(self.Npara):
            self.dphilist.append(dphi[:,k])
        return self.dphilist
            
    def grad_phix(self, x):
        self.grad_phix_list = []
        for k in range(self.Npara):
            dphi = self.dphilist[k]
            dphix = self.fem.eval_phix(x, dphi)
            self.grad_phix_list.append(dphix)
        return self.grad_phix_list

        
    def grad_R(self):
        self.grad_R_list = []
        
        x_index, vh_index = self.fem.get_boundary_index(self.fem.labelleft)
        nt = vh_index.shape[0]
        
        self.mode_pattern = np.ones(nt, dtype = complex)
        vertices = self.fem.mesh.vertices
        XTCOORX = np.vstack((vertices[x_index[:,0], 0], vertices[x_index[:,1], 0])).transpose()
        XTCOORY = np.vstack((vertices[x_index[:,0], 1], vertices[x_index[:,1], 1])).transpose()
        XTCOOR = [XTCOORX, XTCOORY]
        ls = np.sqrt((XTCOOR[0][:,0]-XTCOOR[0][:,0])**2+(XTCOOR[1][:,1]-XTCOOR[1][:,0])**2)
        ls = np.expand_dims(ls, axis = 1)
        self.boundaryL = np.sum(ls)
        if self.ord == 'P1':
            intmat = np.array([1/2, 1/2], dtype = np.float64)
        elif self.ord == 'P2':
            intmat = np.array([1/6, 2/3, 1/6], dtype = np.float64)
        for k in range(self.Npara):
            dphi = self.dphilist[k]
            dphi_related = dphi[vh_index.flatten()].reshape(nt, self.fem.dofedge)
            intdphi = dphi_related * intmat * ls
            intdphimode = np.sum(intdphi, axis = 1) * self.mode_pattern
            
            mode_match = np.sum(intdphimode)/self.boundaryL
            self.grad_R_list.append(mode_match / self.fem.H0)
        return self.grad_R_list
        
            
    def grad_R2(self):
        self.grad_R2_list = []
        for n in range(len(self.grad_R)):
            grad_R2 = self.grad_R[n]*self.fem.R.conj()+self.grad_R[n].conj()*self.fem.R
            self.grad_R2_list.append(grad_R2.real.astype(float) )
        return self.grad_R2_list
    

def Uinc(x, k0):
    f = torch.exp(-1j*k0*x[0])
    return f

def area(x,y):
    return torch.abs((x[:,0]*(y[:,1]-y[:,2])+x[:,1]*(y[:,2]-y[:,0])+x[:,2]*(y[:,0]-y[:,1]))/2.)

def isInside(x, XTCOOR):
    err = 1e-12
    nt = XTCOOR[0].shape[0]
    A1 = torch.zeros(nt, dtype = torch.float64, device = XTCOOR[0].device)
    # XTCOOR = [Xcoor, Ycoor], Xcoor[nt, 3] = [point0[1:nt], point1[1:nt], point2[1:nt]]
    A = area(XTCOOR[0], XTCOOR[1])
    for k in range(3):
        XTCOORx_temp = torch.clone(XTCOOR[0])
        XTCOORy_temp = torch.clone(XTCOOR[1])
        XTCOORx_temp[:,k] = x[0]
        XTCOORy_temp[:,k] = x[1]
        A1 += area(XTCOORx_temp, XTCOORy_temp)
    element_number = torch.nonzero(torch.abs(A1-A)<err)
    element_number = element_number.cpu().detach().numpy().flatten()
    return element_number[0].astype(int)


