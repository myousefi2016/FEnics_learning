from dataclasses import dataclass
from fenics import *
from numpy import arctan, array, power
import matplotlib.pyplot as plt

@dataclass
class caseData:
    mesh : Mesh = Mesh()

    inletLabel  : int = 1
    outletLabel : int = 2
    wallLabel   : int = 3

    frictionVelocity : float = 5.43496e-2
    mu               : float = 1e-4
    beta             : float = 3/40
    betaStar         : float = 0.09
    sigma            : float = 0.5
    sigmaStar        : float = 0.5
    gamma            : float = 5/9


def meshGenerate(caseData):
    caseData.mesh = UnitIntervalMesh(198)
    yCoordinate = caseData.mesh.coordinates()
    yCoordinate[:] = yCoordinate[:]*2-1
    yCoordinate[yCoordinate<0] = (power(1+550, yCoordinate[yCoordinate<0]+1)-1)/550 - 1
    yCoordinate[yCoordinate>0] = -(power(1+550, 1-yCoordinate[yCoordinate>0])-1)/550 + 1

channelFlow = caseData()
meshGenerate(channelFlow)

def boundary(x, on_boundary):
    return on_boundary

V = FunctionSpace(channelFlow.mesh, 'CG', 1)
u_test = TestFunction(V)
u = TrialFunction(V)
u_n = Function(V)
u_ = Function(V)

P = FiniteElement('CG', interval, 1)
element = MixedElement([P, P])
T = FunctionSpace(channelFlow.mesh, element)

k_test, w_test = TestFunctions(T)
k, w = TrialFunctions(T)

T_n = Function(T)
k_n, w_n = T_n.split()
T_ = Function(T)
k_, w_ = T_.split()

u_ = project(Constant("0.0"), V)
k_ = project(Constant("0.0"), V)
w_ = project(Constant("1e5"), V)

gradP = Constant(channelFlow.frictionVelocity**2)

mu = Constant(channelFlow.mu)
beta = Constant(channelFlow.beta)
betaStar = Constant(channelFlow.betaStar)
sigma = Constant(channelFlow.sigma)
sigmaStar = Constant(channelFlow.sigmaStar)
gamma = Constant(channelFlow.gamma)

relax = 0.1

F1 = (mu+k_/w_)*dot(grad(u), grad(u_test))*dx - gradP*u_test*dx
F2 = (k_/w_)*dot(dot(grad(u_), grad(u_)), k_test)*dx - betaStar*k_*w_*k_test*dx + (mu + sigmaStar*k_/w_)*dot(grad(k), grad(k_test))*dx
F3 = gamma*dot(dot(grad(u_), grad(u_)), w_test)*dx - beta*k_*w_*w_test*dx + (mu + sigma*k_/w_)*dot(grad(w), grad(w_test))*dx

F = F2 + F3

bc_u = DirichletBC(V, Constant(0.0), boundary)
bc_k = DirichletBC(T.sub(0), Constant(0.0), boundary)
bc_w = DirichletBC(T.sub(1), Constant(2.35e14), boundary)

a1 = lhs(F1)
L1 = rhs(F1)

a2 = lhs(F)
L2 = rhs(F)

for i in range(1):
    A1 = assemble(a1)
    b1 = assemble(L1)

    A2 = assemble(a2)
    b2 = assemble(L2)
 
    bc_u.apply(A1)
    bc_u.apply(b1)
    bc_k.apply(A2)
    bc_k.apply(b2)
    bc_w.apply(A2)
    bc_w.apply(b2)

    solve(A1, u_.vector(), b1)
    solve(A2, T_.vector(), b2)
    print(u_(0))
    print(T_.sub(0)(0))
    print(T_.sub(1)(0))

    # Un = project(0.99*Un + 0.01*Un_, U)
    # print(Un.sub(2)(0))