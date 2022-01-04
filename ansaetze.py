from discopy import Ty
from discopy.quantum import Circuit, sqrt, Ket, Bra, H, Rx, CX
from discopy.quantum.circuit import Functor
from discopy.rigid import Spider

# ansatz for transitive verbs
def transitive_ansatz(phase):
    return sqrt(2) @ Ket(0, 0) >> H @ Rx(phase) >> CX

# ansatz for relative pronouns
GHZ = Functor(ob=lambda o: o, ar=lambda a: a)(sqrt(2) @ Spider(0, 3, Ty('n')))

# ansatz for cups
cup = Functor(ob=lambda o: o, ar=lambda a: a)(CX >> H @ Circuit.id(1) >> Bra(0, 0) @ sqrt(2))
