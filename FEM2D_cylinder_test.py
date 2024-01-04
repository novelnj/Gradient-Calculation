# FEM2D_vectorP2_final, 102223
import numpy as np
import torch
import time
from READMESH import *
from FEM2D_vectorP2 import *
PI = np.pi

meshfile = r'freefem\export_mesh.msh'
mesh2d = MESH(meshfile)
vh = np.loadtxt(r'freefem\export_vh.txt', dtype = int)
print('# of vh entries is: ', vh.shape[0])
print('ndof is: ', np.max(vh)+1)

ur = np.loadtxt(r'freefem\ureal.txt', dtype = np.float64)
ui = np.loadtxt(r'freefem\uimag.txt', dtype = np.float64)
Kr = np.loadtxt(r'freefem\K_matrix1.txt', skiprows=3, dtype = np.float64)
Ki = np.loadtxt(r'freefem\K_matrix2.txt', skiprows=3, dtype = np.float64)
br = np.loadtxt(r'freefem\b_1.txt', dtype = np.float64)
bi = np.loadtxt(r'freefem\b_2.txt', dtype = np.float64)
phi_freefem = ur+1j*ui
b_freefem = br+1j*bi
K_freefem = Kr[:,2] + 1j*Ki[:,2]
K_freefem = np.vstack((Kr[:,0], Kr[:,1], K_freefem))
print(K_freefem.shape)



mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(mydevice)
t = time.time()
fem = FEM2D(mesh2d, vh, 'P2', mydevice, solver = 'default')
K_flat = fem.calc_K()
KSE = fem.calc_B()

phi = fem.solve_system()
#tempK = fem.sparseK.todense()
#tempb = fem.sparseb.todense()
x = torch.tensor([-2*0.3,0], requires_grad = False, device = mydevice)
phix = fem.eval_phix(x, phi)
print('Self sparseK norm is: ', np.linalg.norm(fem.sparseK.data, ord = 2))
print('Self sparseb norm is: ', np.linalg.norm(fem.sparseb.data, ord = 2))

phifem = fem.solve_system(K_freefem, b_freefem)
phixfem = fem.eval_phix(x, phifem)

print('FREEFEM++ sparseK norm is: ', np.linalg.norm(fem.sparseK.data, ord = 2))
#print('Delta sparseK norm is: ', np.linalg.norm(tempK-fem.sparseK.todense(), ord = 'fro'))

print('FREEFEM++ sparseb data norm is: ', np.linalg.norm(fem.sparseb.data, ord = 2))
#print('Delta sparseb norm is: ', np.linalg.norm(tempb-fem.sparseb.todense(), ord = 2))

print('Self Reflection coefficient is: ', phix)
print('FREEFEM++ Reflection coefficient is: ', phixfem)

print('Delta PHI norm is: ', np.linalg.norm(phi-phifem, ord = 2))


t = time.time()
para = torch.tensor([0., 0.], requires_grad = True, device = mydevice)
case = ['geor', 'eps']
label = [5, 0]
grad2d = grad(fem, solver = 'default')

grad2d.grad_jac(para, case = case, label = label)
print('Jacobian matrix construction time is: ', time.time()-t)

t = time.time()
grad2d.grad_phi()
grad_phix_list = grad2d.grad_phix(x)
#dR = grad2d.grad_R()
#dR2 = grad2d.grad_R2()
print('AVM solver time is: ', time.time()-t)

print('grad_phix is: ', grad_phix_list )
#print('grad_R is: ', grad2d.grad_R)
#print('grad_|R|^2 is: ', grad2d.grad_R2)
pass
#print('|R|^2 is: ', R2.real)