import ast
import json
import numpy as np
import operator,os,sys,subprocess

def main():

    # load in TCIT result
    G4_list   = []
    TCIT_list = []

    # Set TCIT dictionary for interested props
    TCIT_Cv = {}
    TCIT_S0 = {}
    TCIT_Gf = {}
    TCIT_Hf = {}

    with open("CvandS.log","r") as f:

        for lc,lines in enumerate(f):
            
            fields = lines.split()

            if len(fields) > 10 and 'can not use TCIT to calculate,' in lines:
                G4_list += [fields[0]]

            if len(fields) == 8 and 'Prediction of Gf_298 for' in lines and fields[4] not in G4_list: 
                TCIT_Gf[fields[4]] = float(fields[-2])

            if len(fields) == 8 and 'Prediction of Hf_298 for' in lines and fields[4] not in G4_list: 
                TCIT_Hf[fields[4]] = float(fields[-2])

            if len(fields) == 8 and 'Prediction of S0_gas for' in lines and fields[4] not in G4_list: 
                TCIT_S0[fields[4]] = float(fields[-2])

            if len(fields) == 8 and 'Prediction of Cv_gas for' in lines and fields[4] not in G4_list: 
                TCIT_Cv[fields[4]] = float(fields[-2])

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

    ## Compare Exp with TCIT
    Exp_TCIT_S0_diff = []
    for smiles in TCIT_S0.keys():
    
        if smiles in Exp_S0.keys():
            Exp_TCIT_S0_diff += [ TCIT_S0[smiles] - np.mean(Exp_S0[smiles])]
            with open("TCIT_Exp_S0.txt",'a') as f:
                f.write("{:<60s} {:<10.2f} {:<10s} {:<10.2f}\n".format(smiles,TCIT_S0[smiles],str(Exp_S0[smiles]),TCIT_S0[smiles]-np.mean(Exp_S0[smiles])))
                
    Exp_TCIT_S0_diff = np.array(Exp_TCIT_S0_diff)
    print(np.mean(Exp_TCIT_S0_diff ))
    print(np.mean(abs(Exp_TCIT_S0_diff )))

    Exp_TCIT_Cv_diff = []
    for smiles in TCIT_Cv.keys():

        if smiles in Exp_Cv.keys():
            Exp_TCIT_Cv_diff += [ TCIT_Cv[smiles] - np.mean(Exp_Cv[smiles])]
            with open("TCIT_Exp_Cv.txt",'a') as f:
                f.write("{:<60s} {:<10.2f} {:<10s} {:<10.2f}\n".format(smiles,TCIT_Cv[smiles],str(Exp_Cv[smiles]),TCIT_Cv[smiles]-np.mean(Exp_Cv[smiles])))
            
    Exp_TCIT_Cv_diff = np.array(Exp_TCIT_Cv_diff)
    print(np.mean(Exp_TCIT_Cv_diff ))
    print(np.mean(abs(Exp_TCIT_Cv_diff )))
    
    
    exit()
    ## Compare Exp with G4
    for smiles in TCIT_Cv.keys():

        if smiles in G4_dict.keys():
            G4_TCIT_Cv_diff += [ TCIT_Cv[smiles]-G4_dict[smiles] ]
            Exp_TCIT_Cv_dev  += [ abs(TCIT_Cv[smiles] - Exp_Cv[smiles]) / Exp_Cv[smiles]] 
            TCIT += [TCIT_Cv[smiles]]
            #with open("cyclic_TCIT_G4_Cv_1.txt",'a') as f:
            #    f.write("{:<60s} {:<10.2f} {:<10.2f} {:<10.2f}\n".format(smiles,TCIT_Cv[smiles],G4_dict[smiles],TCIT_Cv[smiles]-G4_dict[smiles]))

    G4_TCIT_Cv_diff = np.array(G4_TCIT_Cv_diff)
    print(max(TCIT),min(TCIT))
    print(np.mean(G4_TCIT_Cv_diff ))
    print(np.mean(abs(G4_TCIT_Cv_diff )))

    #################### Compare TCIT with Exp ##############

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

if __name__ == "__main__":
    main()

