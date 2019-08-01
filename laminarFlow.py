from dataclasses import dataclass
import numpy as np
import fenics as fe

@dataclass
class caseData:
    Lx : float = -10
    Ly : float = 8
    nx : int = 40
    ny : int = 30

def meshGeneration(caseData):
    caseData.mesh = fe.RectangleMesh(fe.Point(0.0, -0.5*caseData.Ly), fe.Point(caseData.Lx, 0.5*caseData.Ly), caseData.nx, caseData.ny)
    
    caseData.inlet = fe.AutoSubDomain(lambda x, on_boundary: fe.near(x[0], 0.0) and on_boundary)
    caseData.outlet = fe.AutoSubDomain(lambda x, on_boundary: fe.near(x[0], caseData.Ly) and on_boundary)
    caseData.wall = fe.AutoSubDomain(lambda x, on_boundary: fe.near(abs(x[1]), 0.5*caseData.Ly) and on_boundary)

    caseData.boundaries = fe.MeshFunction("size_t", caseData.mesh, caseData.mesh.topology().dim()-1, 0)
    caseData.ds = fe.ds(subdomain_data = caseData.boundaries)

    caseData.velocity = fe.VectorFunctionSpace(caseData.mesh, 'CG', 2)
    caseData.pressure = fe.FunctionSpace(caseData.mesh, 'CG', 1)

    v_elem = fe.VectorElement("CG", caseData.mesh.ufl_cell(), 1, 2)
    p_elem = fe.FiniteElement("CG", caseData.mesh.ufl_cell(), 1)
    caseData.MixedSpace = fe.FunctionSpace(caseData.mesh, fe.MixedElement([v_elem, p_elem]))

    caseData.bc_wall = fe.DirichletBC(caseData.MixedSpace.sub(caseData.velocity), fe.Constant(0.0), caseData.boundaries)

if __name__=="__main__":
    laminarJet = caseData()
    meshGeneration(laminarJet)



