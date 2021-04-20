import numpy as np
import pandas as pd
import ast,operator,os,sys,subprocess
import json
from copy import deepcopy

def main():

    # load in TCIT result
    G4_list   = ["FS(F)(F)F","FS(F)(F)(F)(F)F","FS(F)(F)(F)(F)Cl"]
    
    # Set TCIT dictionary for interested props
    TCIT_result= {}
    G4_result  = {}

    # suspect list: exp/G4 might be wrong
    S0_suspect = ["CCNc1ccccc1","CSc1ccccc1","CCN1CCOCC1","CCO[S](=O)(=O)OCC","CCOC(=O)C(=O)OCC","CCCCCCCCCl","CCCCCCCCCCC#N","CCCCCCCCCCCC#N","OC1CCCC1","C[C@@H]1CC[C@H]1C","Cc1ccc(Cl)cc1"]
    Cv_suspect = ["CC(C)CCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCC(C)C","CCO[S](=O)(=O)OCC","CC(C)CCCCCOC(=O)c1ccccc1C(=O)OCCCCCC(C)C","CCCCOC(=O)c1ccccc1C(=O)OCCCC","CCOC(=O)c1ccccc1C(=O)OCC",\
                  "CCCCCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCCCC","CCCCCCOC(=O)c1ccccc1C(=O)OCCCCCC","CCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCC","CCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCC",\
                  "CC(C)CCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCC(C)C","CCCOC(=O)c1ccccc1C(=O)OCCC","CC(C)COC(=O)c1ccccc1C(=O)OCC(C)C","COC(=O)c1ccc(cc1)C(=O)OC","CCCCCCCCOC(=O)c1ccc(cc1)C(=O)OCCCCCCCC"]

    # parse result
    with open("paper.log","r") as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if len(fields) > 10 and 'can not use TCIT to calculate,' in lines and fields[0] not in G4_list:
                G4_list += [fields[0]]
    
    with open("paper_output.txt","r") as f:
        for lc,lines in enumerate(f):
            if lc == 0: continue
            fields = lines.split()
            if len(fields) == 0 or fields[0] == '#': continue

            if fields[0] in G4_list and fields[0] not in G4_result.keys():
                G4_result[fields[0]] = {}
                G4_result[fields[0]]["S0"],G4_result[fields[0]]["Cv"],G4_result[fields[0]]["Gf"] = float(fields[4]),float(fields[5]),float(fields[3])

            if fields[0] not in G4_list and fields[0] not in TCIT_result.keys():
                TCIT_result[fields[0]] = {}
                TCIT_result[fields[0]]["S0"],TCIT_result[fields[0]]["Cv"],TCIT_result[fields[0]]["Gf"] = float(fields[4]),float(fields[5]),float(fields[3])
    
    # load in G4 dict
    G4_dict  = parse_G4_database('/'.join(os.path.abspath(__file__).split('/')[:-3])+'/database/G4_thermo.db')
    similar_match = {}
    for inchi in G4_dict.keys():
        similar_match[inchi[:14]]=inchi
    
    # load in Exp dict
    with open("Cv_data.json","r") as f:
        Exp_Cv = json.load(f)

    with open("entropy_data.json","r") as f:
        Exp_S0 = json.load(f)

    # load in smi2inchi
    with open("smiles_to_inchikey.json","r") as f:
        S2I = json.load(f)

    ## Compare Exp with TCIT
    data = {}
    for smiles in TCIT_result.keys():

        data[smiles]={}
        data[smiles]["S0_TCIT"] = TCIT_result[smiles]["S0"]
        data[smiles]["Cv_TCIT"] = TCIT_result[smiles]["Cv"]
        data[smiles]["Gf_TCIT"] = TCIT_result[smiles]["Gf"]

        if smiles in Exp_S0.keys() and smiles not in S0_suspect:
            data[smiles]["S0_Exp"] = np.mean(Exp_S0[smiles])

        if smiles in Exp_Cv.keys() and smiles not in Cv_suspect:
            data[smiles]["Cv_Exp"] = np.mean(Exp_Cv[smiles])

        inchikey = S2I[smiles]
        if inchikey not in G4_dict.keys() and inchikey[:14] in similar_match.keys():
            inchikey = similar_match[inchikey[:14]]

        if inchikey in G4_dict.keys():
            data[smiles]["S0_G4"] = G4_dict[inchikey]["S0"]/0.239006
            data[smiles]["Gf_G4"] = G4_dict[inchikey]["GF_298"]/0.239006
            Cv_corr = return_Cv_corr(smiles)
            data[smiles]["Cv_G4"] = G4_dict[inchikey]["Cv"]/0.239006+Cv_corr

            # also append data to G4_result
            G4_result[smiles]={}
            G4_result[smiles]["S0"]=G4_dict[inchikey]["S0"]/0.239006
            G4_result[smiles]["Cv"]=G4_dict[inchikey]["Cv"]/0.239006+Cv_corr
            
    df=pd.DataFrame.from_dict(data, orient='index')
    df.to_csv('TCIT_result.csv')

    G4data = {}
    for smiles in G4_result.keys():
        G4data[smiles]={}
        G4data[smiles]["S0_G4"] = G4_result[smiles]["S0"]
        G4data[smiles]["Cv_G4"] = G4_result[smiles]["Cv"]
        if smiles in Exp_S0.keys() and smiles not in S0_suspect: 
            G4data[smiles]["S0_Exp"] = np.mean(Exp_S0[smiles])

        if smiles in Exp_Cv.keys() and smiles not in Cv_suspect:
            G4data[smiles]["Cv_Exp"] = np.mean(Exp_Cv[smiles])

    df=pd.DataFrame.from_dict(G4data, orient='index')
    df.to_csv('G4_result.csv')
    
    return

# Description: Simple wrapper function for writing xyz file                           
#
# Inputs      name:     string holding the filename of the output                                                     
#             elements: list of element types (list of strings)                                             
#             geo:      Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
# Returns     None                                                           
#                                                                               
def xyz_write(Output,Elements,Geometry,charge=0):
    
    # Open file for writing and write header
    fid = open(Output,'w')
    fid.write('{}\n'.format(len(Elements)))
    fid.write('q {}\n'.format(charge))
    for count_i,i in enumerate(Elements):
        fid.write('{: <4} {:< 12.6f} {:< 12.6f} {}\n'.format(i,Geometry[count_i,0],Geometry[count_i,1],Geometry[count_i,2]))

    fid.close()

# load in G4 database
def parse_G4_database(db_files,G4_dict={}):
    with open(db_files,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0] == '#': continue
            if len(fields) >= 7:
                if fields[0] not in G4_dict.keys():
                    G4_dict[fields[0]] = {}
                    G4_dict[fields[0]]["smiles"] = fields[1]
                    G4_dict[fields[0]]["HF_0"]   = float(fields[2])
                    G4_dict[fields[0]]["HF_298"] = float(fields[3])
                    G4_dict[fields[0]]["S0"]     = float(fields[4])
                    G4_dict[fields[0]]["GF_298"] = float(fields[5])
                    G4_dict[fields[0]]["Cv"]     = float(fields[6])

    return G4_dict

# Function that take smile string and return element and geometry
def parse_smiles(smiles):

    # load in rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # construct rdkir object
    m = Chem.MolFromSmiles(smiles)
    m2= Chem.AddHs(m)
    AllChem.EmbedMolecule(m2)
    
    # parse mol file and obtain E & G
    lines = Chem.MolToMolBlock(m2).split('\n')
    E = []
    G = []
    for line in lines:
        fields = line.split()
        if len(fields) > 5 and fields[0] != 'M' and fields[-1] != 'V2000':
            E  += [fields[3]]
            geo = [float(x) for x in fields[:3]]
            G  += [geo]

    G = np.array(G)
    return E,G

# function to determine number of rotatable bonds
def return_Nrot(E,G):
    
    xyz_write("Nrot_input.xyz",E,G)
    # use obconformer to determine the functional group
    substring = "obconformer 1 1 Nrot_input.xyz"
    output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[1] 
    output = output.decode('utf-8').split('\n')
    Nrot=0
    for line in output:
        fields = line.split()
        if len(fields) == 5 and fields[0] == 'NUMBER' and fields[2] == 'ROTATABLE':
            try:
                Nrot = int(fields[-1])
            except:
                print("Have error using obconformer to determine N_rot, Nrot = 0")
                
    # Remove the tmp file that was read by obconformer
    try:
        os.remove('Nrot_input.xyz')
    except:
        pass

    return Nrot

# determine Cv correction
def return_Cv_corr(smiles):

    E,G = parse_smiles(smiles)
    Nrot = return_Nrot(E,G)

    return 1.95 * Nrot + 0.84

if __name__ == "__main__":
    main()

