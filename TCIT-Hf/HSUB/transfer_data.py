import json
# Function to call rdkit to transfer smiles into inchikey    
def smiles2inchikey(smiles):

    # load in rdkit
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdinchi import MolToInchiKey

    return MolToInchiKey(MolFromSmiles(smiles))

Hsub_dict={}
with open('HSUB_performance.txt','r') as f:
    for lc,lines in enumerate(f):
        if lc == 0: continue
        fields = lines.split()
        smiles = fields[0]
        try:
            inchikey = smiles2inchikey(smiles)
            Exp      = float(fields[2])
            Hsub_dict[inchikey] = Exp

        except:
            print("Invlid smiles: {}".format(smiles))
            pass

with open("NIST_HSUB.json", "w") as fp:
    json.dump(Hsub_dict, fp)

        
