smiles = []
with open("Solid_compare.txt","r") as f:
    for lc,lines in enumerate(f):
        fields = lines.split()
        smiles += [fields[0]]

from rdkit import Chem
import numpy as np
Nheavy = [Chem.MolFromSmiles(i).GetNumAtoms() for i in smiles]
print(len([i for i in Nheavy if i > 12]))
print(np.mean(Nheavy))

smiles = []
with open("Liquid_compare.txt","r") as f:
    for lc,lines in enumerate(f):
        fields = lines.split()
        smiles += [fields[0]]

from rdkit import Chem
import numpy as np
Nheavy = [Chem.MolFromSmiles(i).GetNumAtoms() for i in smiles]
print(len([i for i in Nheavy if i > 12]))
print(np.mean(Nheavy))
