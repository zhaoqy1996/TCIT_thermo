import sys,os,argparse,subprocess,shutil,time,glob,fnmatch
import numpy as np
from rdkit import Chem
from copy import deepcopy

def main():

    smiles="C1=CC=C(C=C1)O"

    P,H_vap = calculate_PandHvap(smiles,T=298,NIST_db='NIST_Hvap.json')
    print(P,H_vap)


# Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix 
def graph_seps(adj_mat_0):

    # Create a new name for the object holding A**(N), initialized with A**(1)
    adj_mat = deepcopy(adj_mat_0)
    
    # Initialize an array to hold the graphical separations with -1 for all unassigned elements and 0 for the diagonal.
    seps = np.ones([len(adj_mat),len(adj_mat)])*-1
    np.fill_diagonal(seps,0)

    # Perform searches out to len(adj_mat) bonds (maximum distance for a graph with len(adj_mat) nodes
    for i in np.arange(len(adj_mat)):        

        # All perform assignments to unassigned elements (seps==-1) 
        # and all perform an assignment if the value in the adj_mat is > 0        
        seps[np.where((seps==-1)&(adj_mat>0))] = i+1

        # Since we only care about the leading edge of the search and not the actual number of paths at higher orders, we can 
        # set the larger than 1 values to 1. This ensures numerical stability for larger adjacency matrices.
        adj_mat[np.where(adj_mat>1)] = 1
        
        # Break once all of the elements have been assigned
        if -1 not in seps:
            break

        # Take the inner product of the A**(i+1) with A**(1)
        adj_mat = np.dot(adj_mat,adj_mat_0)

    return seps

# smarts for functional groups
def identify_functional_groups(smi):
    '''Identify the presence of functional groups present in molecule                                                                                                  
       denoted by smiles string
                                                     
    Returns:                                                                                                                                                           
        mol_func_groups: (list) contains binary values of functional groups presence                                                                                   
                          None if inchi to molecule conversion returns warning or error                                                                                
    '''
    # smart string for all function groups (https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html)
    func_grp_smarts = {'alkene':'[CX3]=[CX3]','peroxide':'[OX2,OX1-][OX2,OX1-]','hydroperoxide':'[OX2H][OX2]','alcohols':'[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]',
                       'amines':'[NX3;H2,H1;!$(NC=O)]', 'nitrate':'[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]',
                       'aromatics':'[$([cX3](:*):*),$([cX2+](:*):*)]','carboxylic acids':'[CX3](=O)[OX2H1]', 'ether': '[OD2;!$(OC~[!#1!#6])]([#6])[#6]',
                       'esters':'[#6][CX3](=O)[OX2H0][#6]', 'ketones':'[#6][CX3](=O)[#6]','aldehydes':'[CX3H1](=O)[#6]','phenol': '[c][OX2H]',
                       'AmidePrimary':'[CX3;$([R0][#6]),$([H1R0])](=[OX1])[NX3H2]','AmideSecondary':'[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3H1][#6;!$(C=[O,N,S])]',
                       'AmideTertiary': '[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3H0]([#6;!$(C=[O,N,S])])[#6;!$(C=[O,N,S])]','nitro':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
                       'amides': '[CX3;$([R0][#6]),$([H1R0])](=[OX1])[#7X3H2,#7X3H1,#7X3H0]'}

    func_grp_structs = {func_name : Chem.MolFromSmarts(func_smarts) for func_name, func_smarts in func_grp_smarts.items()}

    func_dict = {}
    
    #try:

    #Convert smiles to molecule
    mol  = Chem.MolFromSmiles(smi)
    atoms= [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    N_C  = atoms.count(6)
    # obtain the adjacency matrix and graph seperation
    adj_mat=Chem.rdmolops.GetAdjacencyMatrix(mol)
    gs = graph_seps(adj_mat)
    
    #populate the list with binary values                                                                                                                          
    for func_name, func_struct in func_grp_structs.items():        
        func_dict[func_name] = mol.GetSubstructMatches(func_struct)
    
    # after all those "seperate" functional groups are identified, transfer into SIMPOL format
    # initialzie functional list
    func_list = [0] * 30
    
    # generate a list of aromatic atoms
    aromatics = [ind[0] for ind in func_dict['aromatics']]
    
    # group 1: number of carbon atoms
    func_list[0] = N_C
        
    # group 2: carbon number on the acid-side of an amide (asa)
    if len(func_dict['amides']) > 0:

        for amide in func_dict['amides']:
            amide_E=[mol.GetAtomWithIdx(ind).GetAtomicNum() for ind in amide]
            N_ind = amide[amide_E.index(7)]
            C_ind = amide[amide_E.index(6)]

            # if the distance from C is less than that from N, this carbon atom is on the asa
            carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetIdx() not in amide]
            func_list[1] += len([ind for ind in carbon_atoms if gs[C_ind][ind] < gs[N_ind][ind] ])

    # group 3: aromatic ring
    func_list[2] = int(len(func_dict['aromatics']) / 6)

    # group 4: non-aromatic ring
    func_list[3] = Chem.GetSSSR(mol) - int(len(func_dict['aromatics']) / 6)

    # group 5: C=C (non-aromatic)
    func_list[4] = len(func_dict['alkene'])

    # group 6: C=C–C=O in non-aromatic
    if len(func_dict['alkene']) > 0:

        for alkene in func_dict['alkene']:

            for C_ind in alkene:

                atom = mol.GetAtomWithIdx(C_ind)

                # get index for all neighboring carbon atoms
                neighbors = [x.GetIdx() for x in atom.GetNeighbors() if x.GetIdx() not in alkene and x.GetAtomicNum() == 6]
            
                # if this neighbor is 'C' and connect to O by a double bond, this is C=C–C=O
                if len(neighbors) > 0:

                    for NC_ind in neighbors:

                        Natom = mol.GetAtomWithIdx(NC_ind)
                        NN_Oind = [x.GetIdx() for x in Natom.GetNeighbors() if x.GetAtomicNum() == 8]  

                        for NN_O in NN_Oind:
                            if str(mol.GetBondBetweenAtoms(NC_ind,NN_O).GetBondType())=='DOUBLE':
                                func_list[5] += 1
    
    # group 7: hydroxyl (alkyl)
    func_list[6] = len(func_dict['alcohols'])

    # group 8: aldehyde
    func_list[7] = len(func_dict['aldehydes'])    

    # group 9: ketone 
    func_list[8] = len(func_dict['ketones'])
    
    # group 10: carboxylic acid
    func_list[9] = len(func_dict['carboxylic acids'])
    
    # group 12-14: ether, ether (alicyclic) and ether aromatics
    ## If both carbons bonded to the oxygen are not part of an aromatic ring, use b12. If the oxygen is within a non-aromatic ring use b13.
    ## Otherwise, use b14.Examples for dimethylether, use b0, b1, and b12; for tetrahydrofuran,useb0, b1, b4, and b13; for methylphenyl ether,useb0, b1, b3, and b14.
    for ether in func_dict['ether']:

        # get the index of oxygen atom
        O_ind = [ind for ind in ether if mol.GetAtomWithIdx(ind).GetAtomicNum() == 8 ][0]
        C_ind = [ind for ind in ether if ind != O_ind]

        # if Oxygen in a ring, this is group 13; otherwise, this will be group 12/14
        if mol.GetAtomWithIdx(O_ind).IsInRing():
            func_list[12] += 1

        elif len([ind for ind in C_ind if ind in aromatics]) > 0:
            func_list[13] += 1

        else:
            func_list[11] += 1

    # group 15: nitrate
    func_list[14] = len(func_dict['nitrate'])

    # group 16: nitro 
    func_list[15] = len(func_dict['nitro'])

    # group 18-21: amine primary/secondary/tertiary/aromatic
    for amine in func_dict['amines']:

        atom = mol.GetAtomWithIdx(amine[0])

        # get index for all neighbors
        neighbors = [x.GetIdx() for x in atom.GetNeighbors()]
        
        # check whether it is bonded to aromatic carbon
        if len([ind for ind in neighbors if ind in aromatics]) > 0:
            func_list[20] += 1

        elif len(neighbors) == 3:
            func_list[19] += 1

        elif len(neighbors) == 2:
            func_list[18] += 1

        else:
            func_list[17] += 1 

    # group 22: amide primary
    func_list[21] = len(func_dict['AmidePrimary'])    

    # group 23: Amide Secondary
    func_list[22] = len(func_dict['AmideSecondary'])    

    # group 24: Amide Tertiary
    func_list[23] = len(func_dict['AmideTertiary'])    

    # group 25-28: peroxide
    ## Examples: for peroxy propanyl nitrate use b0, b1, and b25; for N-propyl-N-butyl peroxide use b0, b1, and b26; for N-butyl peroxide use b0, b1, and b27;
    ## for peroxyacetic acid use b0, b1, and b28.
    ## Notes: b25 is not common and SMARTS has difficulty identify it, combine this with b28
    for peroxide in func_dict["peroxide"]:

        peroxyacid = []
        
        # loop over two oxygen atoms at first, if one oxygen connects to (C=O), this is Peroxy acid --> b28
        for O_ind in peroxide:

            atom = mol.GetAtomWithIdx(O_ind)

            # get index for all neighbors
            neighbors = [x.GetIdx() for x in atom.GetNeighbors() if x.GetIdx() not in peroxide]
            
            # if this neighbor is 'C' and connect to O by a double bond, this is Peroxy acid --> b28
            if len(neighbors) == 1 and mol.GetAtomWithIdx(neighbors[0]).GetAtomicNum() == 6:

                # find carbon neighbors
                atomC = mol.GetAtomWithIdx(neighbors[0]) 
                C_neighbor = [(x.GetIdx(),x.GetAtomicNum()) for x in atomC.GetNeighbors() if x.GetIdx() not in peroxide] 

                # find whether another O is connect with neigher carbon
                NN_O_ind = [NN[0] for NN in C_neighbor if NN[1] == 8]
                if len(NN_O_ind) > 0 and str(mol.GetBondBetweenAtoms(neighbors[0],NN_O_ind[0]).GetBondType())=='DOUBLE':

                    peroxyacid += [sorted(peroxide)]
                    func_list[27] += 1

    # loop over remaining peroxide:
    total_peroxide= [sorted(i) for i in func_dict["peroxide"]]
    hydroperoxide = [sorted(i) for i in func_dict["hydroperoxide"]]
    
    func_list[25] = len([peroxide for peroxide in total_peroxide if peroxide not in peroxyacid and peroxide not in hydroperoxide])
    func_list[26] = len([peroxide for peroxide in hydroperoxide if peroxide not in peroxyacid])
    
    # group 11: ester & group 30: nitroester
    ### Use with total number of ester groups unless there is a nitro bonded to the acid side carbon chain of the ester, in this case use b30
    nitroester = []   
    if len(func_dict["esters"]) > 0 and len(func_dict["nitro"]) > 0:  

        for ester in func_dict["esters"]:
            # determine the center carbon, which connects to three other atoms
            for ind in ester:
                
                atom = mol.GetAtomWithIdx(ind) 

                # get index for all neighbors
                neighbors = [x.GetIdx() for x in atom.GetNeighbors()]              

                # count how many atoms in "ester" in neighbor list
                if atom.GetAtomicNum() == 6 and len([indi for indi in ester if indi in neighbors]) == 3:
                    CC_ind = ind
                # O1 refers to C-O oxygen, non-acid side of ester
                if atom.GetAtomicNum() == 8 and len(neighbors) == 2:
                    O1_ind = ind

            # loop over all nitro groups, check if there is a nitro bonded to the acid side carbon chain of the ester (within distance<3)
            for nitro in func_dict["nitro"]:

                # identify atom N
                N_ind = nitro[[mol.GetAtomWithIdx(ind).GetAtomicNum() for ind in nitro].index(7)]

                # find the distance between N and CC/O1
                dis1 = gs[CC_ind][N_ind]
                dis2 = gs[O1_ind][N_ind]

                # if dis1 < dis2 (one the acid side) & dis1 <=3, this is nitroester
                if dis1 < dis2 and dis1 <=3:
                    nitroester += [ester]

    func_list[10] = len(func_dict["esters"]) - len(nitroester)
    func_list[29] = len(nitroester)
    
    # group 17: aromatic hydroxyl & group 29: nitrophenol
    if len(func_dict["phenol"]) > 0 and len(func_dict["nitro"]) > 0:

        is_nitrophenol = False
        # loop over all nitro group to see whether it is on aromatic carbon
        for nitro in func_dict["nitro"]:
            
            if len([ind for ind in nitro if ind in aromatics]) > 0:
                is_nitrophenol = True
                break

        if is_nitrophenol:
            func_list[28] = len(func_dict["phenol"])

        else:
            func_list[16] = len(func_dict["phenol"])  

    return func_list


    #except:

     #   return None

def calculate_PandHvap(smiles,T=298.15,NIST_db=''):

    # first check whether Hvap can be found in NIST
    import json
    # load in rdkit
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdinchi import MolToInchiKey

    if os.path.isfile(NIST_db):
        with open(NIST_db,'r') as f:
            NIST = json.load(f)
    
        try:
            inchikey = MolToInchiKey(MolFromSmiles(smiles))
            Hvap = NIST[inchikey]
            return 0,Hvap 

        except:
            pass
        
    # Chemical groups contribution dictionary
    para_dict = {0: {"name": "zeroeth group", "Bk":[-4.26938E+02,2.89223E-01,4.42057E-03,2.92846E-01] },            1: {"name": "carbon number", "Bk": [-4.11248E+02,8.96919E-01,-2.48607E-03,1.40312E-01] },\
                 2: {"name": "carbon number on the asa", "Bk":[-1.46442E+02,1.54528E+00,1.71021E-03,-2.78291E-01]}, 3: {"name": "aromatic ring", "Bk":[3.50262E+01,-9.20839E-01,2.24399E-03,-9.36300E-02]},\
                 4: {"name": "non-aromatic ring", "Bk":[-8.72770E+01,1.78059E+00,-3.07187E-03,-1.04341E-01]},       5: {"name": "C=C (non-aromatic)", "Bk":[5.73335E+00,1.69764E-02,-6.28957E-04,7.55434E-03]},\
                 6: {"name": "C=C-C=O", "Bk":[-2.61268E+02,-7.63282E-01,-1.68213E-03,2.89038E-01]},                 7: {"name": "hydroxyl (alkyl)", "Bk":[-7.25373E+02,8.26326E-01,2.50957E-03,-2.32304E-01]},\
                 8: {"name": "aldehyde", "Bk":[-7.29501E+02,9.86017E-01,-2.92664E-03,1.78077E-01]},                 9: {"name": "ketone", "Bk":[-1.37456E+01,5.23486E-01,5.50298E-04,-2.76950E-01]},\
                 10:{"name": "carboxylic acid", "Bk":[-7.98796E+02,-1.09436E+00,5.24132E-03,-2.28040E-01]},         11:{"name": "ester",  "Bk":[-3.93345E+02,-9.51778E-01,-2.19071E-03,3.05843E-01]},\
                 12:{"name": "ether", "Bk":[-1.44334E+02,-1.85617E+00,-2.37491E-05,2.88290E-01]},                   13:{"name": "ether (alicyclic)", "Bk":[4.05265E+01,-2.43780E+00,3.60133E-03,9.86422E-02]},\
                 14:{"name": "ether, aromatic", "Bk":[-7.07406E+01,-1.06674E+00,3.73104E-03,-1.44003E-01]},         15:{"name": "nitrate", "Bk":[-7.83648E+02,-1.03439E+00,-1.07148E-03,3.15535E-01]},\
                 16:{"name": "nitro", "Bk":[-5.63872E+02,-7.18416E-01,2.63016E-03,-4.99470E-02]},                   17:{"name": "aromatic hydroxyl", "Bk":[-4.53961E+02,-3.26105E-01,-1.39780E-04,-3.93916E-02]},\
                 18:{"name": "amine, primary", "Bk":[3.71375E+01,-2.66753E+00,1.01483E-03,2.14233E-01]},            19:{"name": "amine, secondary", "Bk":[-5.03710E+02,1.04092E+00,-4.12746E-03,1.82790E-01]},\
                 20:{"name": "amine, tertiary", "Bk":[-3.59763E+01,-4.08458E-01,1.67264E-03,-9.98919E-02]},         21:{"name": "amine, aromatic", "Bk":[-6.09432E+02,1.50436E+00,-9.09024E-04,-1.35495E-01]},\
                 22:{"name": "amide, primary", "Bk":[-1.02367E+02,-7.16253E-01,-2.90670E-04,-5.88556E-01]},         23:{"name": "amide, secondary", "Bk":[-1.93802E+03,6.48262E-01,1.73245E-03,3.47940E-02]},\
                 24:{"name": "amide, tertiary", "Bk":[-5.26919E+00,3.06435E-01,3.25397E-03,-6.81506E-01]},          25:{"name": "carbonylperoxynitrate", "Bk":[-2.84042E+02,-6.25424E-01,-8.22474E-04,-8.80549E-02]},\
                 26:{"name": "peroxide", "Bk":[1.50093E+02,2.39875E-02,-3.37969E-03,1.52789E-02]},                  27:{"name": "hydroperoxide", "Bk":[-2.03387E+01,-5.48718E+00,8.39075E-03,1.07884E-01]},\
                 28:{"name": "carbonylperoxyacid", "Bk":[-8.38064E+02,-1.09600E+00,-4.24385E-04,2.81812E-01]},      29:{"name": "nitrophenol", "Bk":[-5.27934E+01,-4.63689E-01,-5.11647E-03,3.84965E-01]},\
                 30:{"name": "nitroester", "Bk":[-1.61520E+03,9.01669E-01,1.44536E-03,2.66889E-01]} }

    func_list = identify_functional_groups(smiles) 
    # calculate P_vap: log_10 P(T) = b_0(T) + \sum v_k,i b_k
    # b_k = B1,k/T + B2,k + B3,kT + B4,klog(T)
    logP = b_0 = para_dict[0]["Bk"][0]/T + para_dict[0]["Bk"][1] + para_dict[0]["Bk"][2]*T + para_dict[0]["Bk"][3]*np.log(T)

    # loop over all functional groups
    for count_k,k in enumerate(func_list):
        # count_k + 1 is ture index
        if k != 0:
            logP += k * (para_dict[count_k+1]["Bk"][0]/T + para_dict[count_k+1]["Bk"][1] + para_dict[count_k+1]["Bk"][2]*T + para_dict[count_k+1]["Bk"][3]*np.log(T) )
        
    ## calculate Delta H_vap: Delta H_vap = 2.303*R*(b'_0(T) + \sum v_k,i b'_k )
    # b'_k = -B1,k+B3,k*T^2+B4,k*T, denote as a rather than b'
    Hvap= a_0 = -para_dict[0]["Bk"][0] + para_dict[0]["Bk"][2]*T**2 + para_dict[0]["Bk"][3]*T

    # loop over all functional groups
    for count_k,k in enumerate(func_list):
        # count_k + 1 is ture index
        if k != 0:
            Hvap += k * (-para_dict[count_k+1]["Bk"][0] + para_dict[count_k+1]["Bk"][2]*T**2 + para_dict[count_k+1]["Bk"][3]*T )
            
    return logP,2.303*8.314*Hvap/1000

if __name__ == "__main__":
    main()
