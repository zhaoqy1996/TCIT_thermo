# This is file is write to 
# 1. Indentify whether this is a minimal structure or not, if so, append it into database
# 2. Use Taffi component increment theory to get the prediction of following properties:
#    A. Enthalpy of formation of gas at 0k and 298k, 1 atom pressure
#    B. Entropy of gas at standard conditions
#    C. Gibbs free energy of formation of gas at standard conditions 
#    D. Constant volumn heat capacity of gas (and constant pressure)
# Author: Qiyuan Zhao

def warn(*args,**kwargs): #Keras spits out a bunch of junk
    pass
import warnings
warnings.warn = warn

import sys,os,argparse,subprocess
import numpy as np
from fnmatch import fnmatch
import tensorflow as tf
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Tensorflow also spits out a bunch of junk
np.random.seed(0)
tf.compat.v2.random.set_seed(0)
random.seed(0)

# import taffi related functions
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/utilities')
from taffi_functions import * 
from deal_ring import get_rings,return_inchikey,return_smi

# import Machine learning related functions
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/ML-package')
import preprocess
import utilities

def main(argv):

    parser = argparse.ArgumentParser(description='This script predicts the enthalpy of formation for given target compounds based '+\
                                                 'on a fixed TCIT CAV database distributed with the paper "A Self-Consistent Component '+\
                                                 'Increment Theory for Predicting Enthalpy of Formation" by Zhao and Savoie. Further ring correction is added' +\
                                                 'distributed with the paper "Ring correction"' +\
                                                 'The script operates on .xyz files/ a list of smiles string, prints the components that it uses for each prediction, and will return the '+\
                                                 '0K and 298K enthalpy of formation, standard entropy and heat ca[acity. '+\
                                                 'Compounds that require missing CAVs are skipped.' )

    #optional arguments                                                                                                                   
    parser.add_argument('-t', dest='Itype', default='xyz',
                        help = 'Controls the input type, either xyz of smiles (default: "xyz")')

    parser.add_argument('-i', dest='input_name',
                        help = 'If input type is xyz, the program loops over all of the .xyz files in this input folder and makes Hf predictions for them (default: "input_xyz",\
                                If input type is smiles, the program loops over all of the smiles string in the given file (default: "input.txt")')

    parser.add_argument('-o', dest='outputname', default='result',
                        help = 'Controls the output file name for the results (default: "results")')

    # parse configuration dictionary (c)                                                                                                   
    print("parsing calculation configuration...")
    args=parser.parse_args()    

    # set default value
    # Energy convert from kj to kcal
    kj2kcal = 0.239006

    if args.Itype not in ['xyz','smiles']:
        print("Warning! input_type must be either xyz or smiles, use default xyz...")
        args.Itype = 'xyz'

    if args.Itype == 'xyz' and args.input_name == None:
        args.input_name = 'input_xyz'

    if args.Itype == 'smiles' and args.input_name == None:
        args.input_name = 'input_list/test_inp.txt'
    
    # load database
    CAV_dict = parse_CAV_database('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/database/TCIT_CAV.db')    
    G4_dict  = parse_G4_database('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/database/G4_thermo.db')
    RC_dict  = parse_ringcorr('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/database/depth0_RC.db')
    
    # create similarity match dictionary 
    # For new conformers which not included in G4 database, find alternative conformer for instead
    similar_match = {}
    for inchi in G4_dict.keys():
        similar_match[inchi[:14]]=inchi
    
    sys.stdout = Logger(args.outputname)

    # find all xyz files in given folder
    if args.Itype == 'xyz':
        target_xyzs=[os.path.join(dp, f) for dp, dn, filenames in os.walk(args.input_name) for f in filenames if (fnmatch(f,"*.xyz"))]
        items      = sorted(target_xyzs)

    else:
        target_smiles = []
        with open(args.input_name,"r") as f:
            for line in f:
                target_smiles.append(line.strip())
        items = target_smiles
        
    # load in ML model
    base_model = getModel()

    # create result dict
    TCITresult = {} 

    # loop for target xyz files
    for i in items:
        print("Working on {}...".format(i))
        if args.Itype == 'xyz':
            E,G = xyz_parse(i)
            adj_mat = Table_generator(E,G)
            name = i.split('/')[-1]
            smiles= return_smi(E,G,adj_mat)

        else:
            E,G = parse_smiles(i)
            adj_mat = Table_generator(E,G)
            name = i
            smiles= i

        if True in [element.lower() not in ['h','c','n','o','f','s','cl','br'] for element in E]:
            print("can't deal with some element in this compounds")
            continue
        
        # check whether this is a ring
        ring_inds = [ring_atom(adj_mat,j) for j,Ej in enumerate(E)]
        
        # obtain number of rotatable bonds for Cv correction (in kJ/mol)
        Nrot = return_Nrot(E,G)
        Cv_corr = 1.95 * Nrot + 0.84

        # determine atom types and replace "R" group by normal group
        atom_types = id_types(E,adj_mat,2)
        atom_types = [atom_type.replace('R','') for atom_type in atom_types]
        
        # remove terminal atoms                                                                                                           
        B_inds = [count_j for count_j,j in enumerate(adj_mat) if sum(j) > 1]
        
        # Apply pedley's constrain 1                                                                                                      
        H_inds = [count_j for count_j in range(len(E)) if E[count_j] == "H" ]
        P1_inds = [ count_j for count_j,j in enumerate(adj_mat) if E[count_j] == "C" and len([ count_k for count_k,k in enumerate(adj_mat[count_j,:]) if k == 1 and count_k not in H_inds ]) == 1 ]
        
        group_types = [atom_types[Bind] for Bind in B_inds if Bind not in P1_inds]
        Unknown = [j for j in group_types if j not in CAV_dict.keys()]
        
        # indentify whether this is a minimal structure or not
        min_types = [ j for j in group_types if minimal_structure(j,G,E,gens=2) is True ]

        # set success predict flag
        success = False
        
        if (len(min_types) > 0 or len(group_types) < 2) and True not in ring_inds:
            print("\n{} can not use TCIT to calculate, the result comes from G4 result".format(name))

        if len(group_types) < 2:
            inchikey = return_inchikey(E,G)
            if inchikey not in G4_dict.keys() and inchikey[:14] in similar_match.keys():
                inchikey = similar_match[inchikey[:14]]
            
            # Look up G4 database 
            if inchikey in G4_dict.keys():
                S0     = G4_dict[inchikey]["S0"]
                Cv     = G4_dict[inchikey]["Cv"]
                Hf_0   = G4_dict[inchikey]["HF_0"]
                Hf_298 = G4_dict[inchikey]["HF_298"]
                Gf_298 = G4_dict[inchikey]["GF_298"]

                print("Prediction of Hf_0 for {} is {} kJ/mol".format(name, Hf_0/kj2kcal))
                print("Prediction of Hf_298 for {} is {} kJ/mol".format(name, Hf_298/kj2kcal))
                print("Prediction of Gf_298 for {} is {} kJ/mol".format(name, Gf_298/kj2kcal))
                print("Prediction of S0 for {} is {} J/(mol*K)".format(name, S0/kj2kcal))
                print("Prediction of Cv for {} is {} J/(mol*K)\n\n".format(name, Cv/kj2kcal+Cv_corr))
                success = True

            else:
                print("G4 calculations are missing for this small compound...")

        else:
            if len(Unknown) == 0: 
                print("\n"+"="*120)
                print("="*120)
                print("\nNo more information is needed, begin to calculate enthalpy of fomation of {}".format(i.split('/')[-1]))
                Hf_0,Hf_298,Gf_298,S0,Cv = calculate_CAV(E,G,adj_mat,name,CAV_dict,RC_dict,base_model,Cv_corr=Cv_corr)
                success = True

            else:
                print("\n"+"="*120)
                print("Unknown CAVs are required for this compound, skipping...\n") 
                print("\n"+"="*120)

        if success:
            TCITresult[name]={}
            TCITresult[name]["Hf_0"]   = Hf_0/kj2kcal
            TCITresult[name]["Hf_298"] = Hf_298/kj2kcal
            TCITresult[name]["Gf_298"] = Gf_298/kj2kcal
            TCITresult[name]["S0"]     = S0/kj2kcal
            TCITresult[name]["Cv"]     = Cv/kj2kcal

    # In the end, write a output file
    with open('{}_output.txt'.format(args.outputname),'w') as f:
        f.write('{:<60s} {:<15s} {:<15s} {:<15s} {:<15s} {:<15s}\n'.format("Molecule","Hf_0k","Hf_298k","Gf_298k","S0_298k","Cv_298k"))
        for i in sorted(TCITresult.keys()):
            f.write('{:<60s} {:< 15.4f} {:< 15.4f} {:< 15.4f} {:< 15.4f} {:< 15.4f}\n'.format(i,TCITresult[i]["Hf_0"],TCITresult[i]["Hf_298"],TCITresult[i]["Gf_298"],TCITresult[i]["S0"],TCITresult[i]["Cv"]))

    return

# function to calculate Hf based on given TCIT database
def calculate_CAV(E,G,adj_mat,name,CAV_dict,RC_dict,base_model,Cv_corr=0):

    # Tabulated  absolute entropy of element in its standard reference state (in J/mol*K)
    # (http://www.codata.info/resources/databases/key1.html)
    S_atom_298k = {"H":65.34, "Li": 29.12, "Be": 9.50, "B": 5.90, "C":5.74, "N":95.805 , "O": 102.58, "F": 101.396, "Na": 51.30, "Mg": 32.67, "Al": 28.30, \
                   "Si": 18.81, "P": 18.81, "S": 32.054, "Cl": 111.54, "Br": 76.11}

    # Energy convert from kj to kcal
    kj2kcal = 0.239006

    # identify ring structure
    ring_inds     = [ring_atom(adj_mat,j) for j,Ej in enumerate(E)] 

    # initialize ring corrections
    ring_corr_Hf0=0
    ring_corr_S0 =0
    ring_corr_Cv =0
    ring_corr_Hf298=0
        
    # Add ring correction to final prediction
    if True in ring_inds:

        # generate depth=0 and depth=2 rings
        try:
            RC0,RC2=get_rings(E,G,gens=2,return_R0=True) 
        except:
            print("ring generation fails for {}, check the geometry of given xyz file or smiles string".format(name))
            
        if len(RC0.keys()) > 0:

            print("Identify rings! Add ring correction to final predictoon")

            for key in RC0.keys():
                depth0_ring=RC0[key]
                depth2_ring=RC2[key] 

                NonH_E = [ele for ele in E if ele is not 'H']
                ring_NonH_E = [ele for ele in depth2_ring["elements"] if ele is not 'H']

                if float(depth0_ring["hash_index"]) in RC_dict.keys() and len(depth2_ring["ringsides"]) == 0:
                    print("\n{} can not use TCIT to calculate, the result comes from G4 result".format(name))
                    RC_Hf0K   = RC_dict[float(depth0_ring["hash_index"])]["HF_0"]
                    RC_Hf298  = RC_dict[float(depth0_ring["hash_index"])]["HF_298"]
                    RC_S0     = RC_dict[float(depth0_ring["hash_index"])]["S_0"]
                    RC_Cv     = RC_dict[float(depth0_ring["hash_index"])]["Cv"]

                    ring_corr_Hf0  +=RC_Hf0K
                    ring_corr_Hf298+=RC_Hf298
                    ring_corr_S0   +=RC_S0
                    ring_corr_Cv   +=RC_Cv
                    
                    print("Add ring correction {}: {:< 6.3f} kcal/mole into final prediction".format(depth0_ring["hash_index"],RC_Hf298))
                    
                elif float(depth0_ring["hash_index"]) in RC_dict.keys() and len(depth2_ring["ringsides"]) > 0:

                    # load in ML weights
                    weights_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])+'/ML-package'
                    base_model.load_weights(weights_path+'/Hf_0k.h5')
                    diff_Hf0K = getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)

                    # predict difference of Hf_298K
                    base_model.load_weights(weights_path+'/Hf_298k.h5')
                    diff_Hf298 = getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)

                    # predict difference of S0
                    base_model.load_weights(weights_path+'/S0_298k.h5')
                    diff_S0 = getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)

                    # predict difference of Cv
                    base_model.load_weights(weights_path+'/Cv.h5')
                    diff_Cv = getPrediction([depth2_ring["smiles"]],[depth0_ring["smiles"]],base_model)

                    # calculate RC based on the differences
                    RC_Hf0K   = RC_dict[float(depth0_ring["hash_index"])]["HF_0"]   + diff_Hf0K
                    RC_Hf298  = RC_dict[float(depth0_ring["hash_index"])]["HF_298"] + diff_Hf298
                    RC_S0     = RC_dict[float(depth0_ring["hash_index"])]["S_0"]    + diff_S0
                    RC_Cv     = RC_dict[float(depth0_ring["hash_index"])]["Cv"]     + diff_Cv

                    ring_corr_Hf0  +=RC_Hf0K 
                    ring_corr_Hf298+=RC_Hf298
                    ring_corr_S0   +=RC_S0 
                    ring_corr_Cv   +=RC_Cv 
                    
                    print("Add ring correction {}: {:< 6.3f} kcal/mole into final prediction (based on depth=0 ring {}:{: < 6.3f})".format(depth2_ring["hash_index"],RC_Hf298,depth0_ring["hash_index"],\
                                                                                                                                           RC_dict[float(depth0_ring["hash_index"])]["HF_298"]))
                else:
                    print("Information of ring {} is missing, the final prediction might be not accurate, please update ring_correction database first".format(depth0_ring["hash_index"]))

        else: 
            print("Identify rings, but the heavy atom number in the ring is greater than 12, don't need add ring correction")

    # determine component types
    atom_types = id_types(E,adj_mat,2)
    atom_types = [atom_type.replace('R','') for atom_type in atom_types]

    # remove terminal atoms                                                                                                           
    B_inds = [count_j for count_j,j in enumerate(adj_mat) if sum(j)>1 ]

    # Apply pedley's constrain 1                                                                                                      
    H_inds = [count_j for count_j in range(len(E)) if E[count_j] == "H" ]
    P1_inds = [ count_j for count_j,j in enumerate(adj_mat) if E[count_j] == "C" and\
                len([ count_k for count_k,k in enumerate(adj_mat[count_j,:]) if k == 1 and count_k not in H_inds ]) == 1 ]
    group_types = [atom_types[Bind] for Bind in B_inds if Bind not in P1_inds]

    # initialize with ring correction 
    Hf_target_0  = ring_corr_Hf0
    Hf_target_298= ring_corr_Hf298
    S0_target    = ring_corr_S0
    Cv_target    = ring_corr_Cv

    for j in group_types:
        S0_target  += CAV_dict[j]["S0"]
        Cv_target  += CAV_dict[j]["Cv"]
        Hf_target_0   += CAV_dict[j]["HF_0"]
        Hf_target_298 += CAV_dict[j]["HF_298"]

    for j in P1_inds:
        NH = len([ind_j for ind_j,adj_j in enumerate(adj_mat[j,:]) if adj_j == 1 and ind_j in H_inds])
        if NH == 3:
            Hf_target_0   += CAV_dict["[6[6[1][1][1]][1][1][1]]"]["HF_0"]
            S0_target     += CAV_dict["[6[6[1][1][1]][1][1][1]]"]["S0"]
            Cv_target     += CAV_dict["[6[6[1][1][1]][1][1][1]]"]["Cv"]
            Hf_target_298 += CAV_dict["[6[6[1][1][1]][1][1][1]]"]["HF_298"]

        elif NH == 2:
            Hf_target_0   += CAV_dict["[6[6[1][1]][1][1]]"]["HF_0"]
            S0_target     += CAV_dict["[6[6[1][1]][1][1]]"]["S0"]
            Cv_target     += CAV_dict["[6[6[1][1]][1][1]]"]["Cv"]
            Hf_target_298 += CAV_dict["[6[6[1][1]][1][1]]"]["HF_298"]

        elif NH == 1:
            Hf_target_0   += CAV_dict["[6[6[1]][1]]"]["HF_0"]
            S0_target     += CAV_dict["[6[6[1]][1]]"]["S0"]
            Cv_target     += CAV_dict["[6[6[1]][1]]"]["Cv"]
            Hf_target_298 += CAV_dict["[6[6[1]][1]]"]["HF_298"]

        else:
            print("Error, no such NH = {} in Constrain 1".format(NH))
            print("{} shouldn't appear here".format([atom_types[Pind] for Pind in P1_inds]))
            quit()
    
    # evaluate Gf based on Hf and S
    S_atom = kj2kcal * sum([ S_atom_298k[_] for _ in E ])
    Gf_target_298 = Hf_target_298 - 298.15 * (S0_target - S_atom) / 1000.0   
    print("Prediction are made based on such group types: (Hf_298k/S0_298k)")
    for j in group_types:
        print("{:30s}: {:<5.2f}/{:<5.2f}".format(j,CAV_dict[j]["HF_298"],CAV_dict[j]["S0"]))

    print("Prediction of Hf_0 for {} is {} kJ/mol".format(name, Hf_target_0/kj2kcal))
    print("Prediction of Hf_298 for {} is {} kJ/mol".format(name, Hf_target_298/kj2kcal))
    print("Prediction of Gf_298 for {} is {} kJ/mol".format(name, Gf_target_298/kj2kcal))
    print("Prediction of S0_gas for {} is {} J/(mol*K)".format(name, S0_target/kj2kcal))
    print("Prediction of Cv_gas for {} is {} J/(mol*K)\n\n".format(name, Cv_target/kj2kcal + Cv_corr))

    return Hf_target_0,Hf_target_298,Gf_target_298,S0_target,Cv_target+Cv_corr*kj2kcal

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

# load in TCIT CAV database
def parse_CAV_database(db_files,CAV_dict={}):
    with open(db_files,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0] == '#': continue
            if len(fields) == 8 and fields[0] not in CAV_dict.keys():
                CAV_dict[fields[0]]  = {}
                CAV_dict[fields[0]]["HF_0"]     = float(fields[1])
                CAV_dict[fields[0]]["HF_298"]   = float(fields[2])
                CAV_dict[fields[0]]["GF_298"]   = float(fields[3])
                CAV_dict[fields[0]]["Cv"]       = float(fields[4])
                CAV_dict[fields[0]]["S0"]       = float(fields[5])
                
    return CAV_dict

# load in ring correction database
def parse_ringcorr(db_file,RC_dict={}):
    with open(db_file,'r') as f:
        for lines in f:
            fields = lines.split()
            if len(fields) ==0: continue
            if fields[0] == '#': continue 
            if len(fields) >= 7:
                RC_dict[float(fields[0])] = {}
                RC_dict[float(fields[0])]["HF_0"]   = float(fields[1])
                RC_dict[float(fields[0])]["HF_298"] = float(fields[2])
                RC_dict[float(fields[0])]["GF_298"] = float(fields[3])
                RC_dict[float(fields[0])]["Cv"]     = float(fields[4])
                RC_dict[float(fields[0])]["S_0"]    = float(fields[5])

    return RC_dict

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

# Description:   Checks if the supplied geometry corresponds to the minimal structure of the molecule
# 
# Inputs:        atomtype:      The taffi atomtype being checked
#                geo:           Geometry of the molecule
#                elements:      elements, indexed to the geometry 
#                adj_mat:       adj_mat, indexed to the geometry (optional)
#                atomtypes:     atomtypes, indexed to the geometry (optional)
#                gens:          number of generations for determining atomtypes (optional, only used if atomtypes are not supplied)
# 
# Outputs:       Boolean:       True if geo is the minimal structure for the atomtype, False if not.
def minimal_structure(atomtype,geo,elements,adj_mat=None,gens=2):

    # If required find the atomtypes for the geometry
    if adj_mat is None:
        if len(elements) != len(geo):
            print("ERROR in minimal_structure: While trying to automatically assign atomtypes, the elements argument must have dimensions equal to geo. Exiting...")
            quit()

        # Generate the adjacency matrix
        # NOTE: the units are converted back angstroms
        adj_mat = Table_generator(elements,geo)

        # Generate the atomtypes
        atom_types= id_types(elements,adj_mat,gens)
        atomtypes = [atom_type.replace('R','') for atom_type in atom_types] 

    # Check minimal conditions
    count = 0
    for count_i,i in enumerate(atomtypes):

        # If the current atomtype matches the atomtype being searched for then proceed with minimal geo check
        if i == atomtype:
            count += 1

            # Initialize lists for holding indices in the structure within "gens" bonds of the seed atom (count_i)
            keep_list = [count_i]
            new_list  = [count_i]
            
            # Carry out a "gens" bond deep search
            for j in range(gens):

                # Id atoms in the next generation
                tmp_new_list = []                
                for k in new_list:
                    tmp_new_list += [ count_m for count_m,m in enumerate(adj_mat[k]) if m == 1 and count_m not in keep_list ]

                # Update lists
                tmp_new_list = list(set(tmp_new_list))
                if len(tmp_new_list) > 0:
                    keep_list += tmp_new_list
                new_list = tmp_new_list
            
            # Check for the minimal condition
            keep_list = set(keep_list)
            if False in [ elements[j] == "H" for j in range(len(elements)) if j not in keep_list ]:
                minimal_flag = False
            else:
                minimal_flag = True
        
    return minimal_flag

#getModel and getPrediction are the two main functions. Build the model and load the parameters,
def getModel():
    params = {
        'n_layers'  :3,
        'n_nodes'   :256,
        'fp_length' :256,
        'fp_depth'  :3,
        'conv_width':30,
        'L2_reg'    :0.0004, #The rest of the parameters don't really do anything outside of training
        'batch_normalization':1,
        'learning_rate':1e-4,
        'input_shape':2
    }

    from gcnn_model import build_fp_model
    
    predictor_MLP_layers = []
    for l in range(params['n_layers']):
        predictor_MLP_layers.append(params['n_nodes'])

    model = build_fp_model(
        fp_length = params['fp_length'],
        fp_depth = params['fp_depth'],
        conv_width=params['conv_width'],
        predictor_MLP_layers=predictor_MLP_layers,
        L2_reg=params['L2_reg'],
        batch_normalization=params['batch_normalization'],
        lr = params['learning_rate'],
        input_size = params['input_shape']
    )

    return model
    
def getPrediction(smiles,R0_smiles,model):
    X_eval = (smiles,R0_smiles)
    processed_eval = preprocess.neuralProcess(X_eval)
    predictions = np.squeeze(model.predict_on_batch(x=processed_eval))
    return predictions

# Logger object redirects standard output to a file.
class Logger(object):
    def __init__(self,filename):
        self.terminal = sys.stdout
        self.log = open("{}.log".format(filename), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

if __name__ == "__main__":
    main(sys.argv[1:])

