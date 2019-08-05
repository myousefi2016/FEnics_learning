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

P = FiniteElement('P', interval, 1)
element = MixedElement([P, P, P])
V = FunctionSpace(channelFlow.mesh, element)

u_test, k_test, w_test = TestFunctions(V)

U_ = Function(V)
U = Function(V)
u_, k_, w_ = split(U_)
u, k, w = split(U)

U_ = project(Expression(("0.0", "0.0", "1e5"), degree = 1), V)

gradP = Constant(channelFlow.frictionVelocity**2)

mu = Constant(channelFlow.mu)
beta = Constant(channelFlow.beta)
betaStar = Constant(channelFlow.betaStar)
sigma = Constant(channelFlow.sigma)
sigmaStar = Constant(channelFlow.sigmaStar)
gamma = Constant(channelFlow.gamma)

relax = 0.1

F = (mu + k_/w_)*dot(grad(u), grad(u_test))*dx - gradP*u_test*dx \
    + (k_/w_)*dot(dot(grad(u_), grad(u_)), k_test)*dx - betaStar*k_*w_*k_test*dx + (mu + sigmaStar*k_/w_)*dot(grad(k), grad(k_test))*dx \
    + gamma*dot(dot(grad(u_), grad(u_)), w_test)*dx - beta*k_*w_*w_test*dx + (mu + sigma*k_/w_)*dot(grad(w), grad(w_test))*dx

bc = DirichletBC(V, (0.0, 0.0, 2.354e14), "on_boundary")

solve(F == 0, U, bc)