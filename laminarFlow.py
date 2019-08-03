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

    caseData.velocity = VectorFunctionSpace(caseData.mesh, 'CG', 2)
    caseData.pressure = FunctionSpace(caseData.mesh, 'CG', 1)

    v_elem = VectorElement("CG", caseData.mesh.ufl_cell(), 1, 2)
    p_elem = FiniteElement("CG", caseData.mesh.ufl_cell(), 1)
    caseData.MixedSpace = FunctionSpace(caseData.mesh, MixedElement([v_elem, p_elem]))

    caseData.bc_wall = DirichletBC(caseData.MixedSpace.sub(caseData.velocity).sub(1), Constant(0.0), caseData.boundaries, 3)

if __name__=="__main__":
    laminarJet = caseData()
    meshGeneration(laminarJet)



