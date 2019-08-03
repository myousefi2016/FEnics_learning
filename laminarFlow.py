from dataclasses import dataclass
from fenics import *
from dolfin import *
from mshr import *
import numpy as np

@dataclass
class caseData:
    Lx : float = -10
    Ly : float = 8
    nx : int = 40
    ny : int = 30

    inletLabel  : int = 1
    outletLabel : int = 2
    wallLabel   : int = 3

def meshGeneration(caseData):
    caseData.mesh = RectangleMesh(Point(0.0, -0.5*caseData.Ly), Point(caseData.Lx, 0.5*caseData.Ly), caseData.nx, caseData.ny)

    caseData.inlet = AutoSubDomain(lambda x, on_boundary: near(x[0], 0.0) and on_boundary)
    caseData.outlet = AutoSubDomain(lambda x, on_boundary: near(x[0], caseData.Ly) and on_boundary)
    caseData.wall = AutoSubDomain(lambda x, on_boundary: near(abs(x[1]), 0.5*caseData.Ly) and on_boundary)

    caseData.boundaries = MeshFunction("size_t", caseData.mesh, caseData.mesh.topology().dim()-1, 0)
    caseData.ds = ds(subdomain_data = caseData.boundaries)

    caseData.inlet.mark(caseData.boundaries, caseData.inletLabel)
    caseData.outlet.mark(caseData.boundaries, caseData.outletLabel)
    caseData.wall.mark(caseData.boundaries, caseData.wallLabel)

    v_elem = VectorElement("CG", caseData.mesh.ufl_cell(), 1, 2)
    v0_elem = FiniteElement("CG", caseData.mesh.ufl_cell(), 1)
    p_elem = FiniteElement("CG", caseData.mesh.ufl_cell(), 1)
    l_elem = FiniteElement("CG", caseData.mesh.ufl_cell(), 1)

    caseData.MixedSpace = FunctionSpace(caseData.mesh, MixedElement([v_elem, p_elem, l_elem]))

    caseData.bc_wall = DirichletBC(caseData.MixedSpace.sub(v_elem).sub(1), Constant(0.0), caseData.boundaries, caseData.wallLabel)
    caseData.bc_lagrange = DirichletBC(caseData.MixedSpace.sub(l_elem), Constant(0.0), "fabs(x[0])>2.0*DOLFIN_EPS")

if __name__=="__main__":
    laminarJet = caseData()
    meshGeneration(laminarJet)



