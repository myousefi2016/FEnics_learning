from dataclasses import dataclass
from fenics import *

@dataclass
class caseData:
    Lx : float = -10
    Ly : float = 8
    nx : int = 40
    ny : int = 30

    inletLabel  : int = 1
    outletLabel : int = 2
    wallLabel   : int = 3

def meshGenerate(caseData):
    caseData.mesh = UnitIntervalMesh(198)
    print(caseData.mesh)

channelFlow = caseData()
meshGenerate(channelFlow)
