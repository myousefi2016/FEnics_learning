from dataclasses import dataclass
from fenics import *
from numpy import arctan, array, power

@dataclass
class caseData:
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
    yCoordinate = caseData.mesh.coordinates()*2-1
    yCoordinate[yCoordinate<0] = (power(1+550, yCoordinate[yCoordinate<0]+1)-1)/550 - 1
    yCoordinate[yCoordinate>0] = -(power(1+550, 1-yCoordinate[yCoordinate>0])-1)/550 + 1

channelFlow = caseData()
meshGenerate(channelFlow)

V = FunctionSpace(channelFlow.mesh, 'P', 1)

u_test = TestFunctions(V)
k_test = TestFunctions(V)
w_test = TestFunctions(V)

u = TrialFunction(V)
k = TrialFunction(V)
w = TrialFunction(V)

# u_ = Function(V)
# k_ = Function(V)
# w_ = Function(V)

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

F = (mu + k_/w_)*grad(u)*grad(u_test)*dx - gradP*u_test*dx 
    #+ (k_/w_)*dot(dot(grad(u_), grad(u_)), k_test)*dx - betaStar*k_*w_*k_test*dx + (mu + sigmaStar*k_/w_)*dot(grad(k), grad(k_test))*dx \
    #+ gamma*dot(dot(grad(u_), grad(u_)), w_test)*dx - beta*k_*w_*w_test*dx + (mu + sigma*k_/w_)*dot(grad(w), grad(w_test))*dx

bc_u = DirichletBC(V, Constant(0.0), "on_boundary")
bc_k = DirichletBC(V, Constant(0.0), "on_boundary")
bc_w = DirichletBC(V, Constant(2.35e14), "on_boundary")

a = lhs(F)
L = rhs(F)

A = assemble(a)
b = assemble(L)

bc_u.apply(A)
bc_u.apply(b)

solve(A, u_.vector(), b)