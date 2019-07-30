from dataclasses import dataclass
import numpy as np
import fenics as fe
from dolfin_adjoint import *

@dataclass
class caseData:
    Lx : float = -10
    Ly : float = 8
    nx : int = 40
    ny : int = 30

def meshGeneration(caseData):
    caseData.mesh = fe.RectangleMesh(fe.Point(0.0, -0.5*caseData.Ly), fe.Point(caseData.Lx, 0.5*caseData.Ly), caseData.nx, caseData.ny)




