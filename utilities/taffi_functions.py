import numpy as np
from scipy.spatial.distance import cdist
from itertools import combinations
from copy import deepcopy
from subprocess import Popen,PIPE
import random,os

# Description: Simple wrapper function for grabbing the coordinates and
#              elements from an xyz file
#
# Inputs      input: string holding the filename of the xyz
# Returns     Elements: list of element types (list of strings)
#             Geometry: Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
def xyz_parse(input,read_types=False):

    # Commands for reading only the coordinates and the elements
    if read_types is False:
        
        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            if len(fields) > 4:
                                Atom_types[count] = fields[4]
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry,Atom_types

# Generates the adjacency matrix based on UFF bond radii
# Inputs:       Elements: N-element List strings for each atom type
#               Geometry: Nx3 array holding the geometry of the molecule
def Table_generator(Elements,Geometry):

    # Initialize UFF bond radii (Rappe et al. JACS 1992)
    # NOTE: Units of angstroms 
    # NOTE: These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist for the same atom, the largest was used. 
    Radii = {  'H':0.354, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.244, 'Si':1.117,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    # SAME AS ABOVE BUT WITH A SMALLER VALUE FOR THE Al RADIUS ( I think that it tends to predict a bond where none are expected
    Radii = {  'H':0.39, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.15,  'Si':1.050,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
               'K':1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.473, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }

    Max_Bonds = {  'H':2,    'He':1,\
                  'Li':None, 'Be':None,                                                                                                                'B':4,     'C':4,     'N':4,     'O':2,     'F':1,    'Ne':1,\
                  'Na':None, 'Mg':None,                                                                                                               'Al':4,    'Si':4,  'P':None,  'S':None, 'Cl':1,    'Ar':1,\
                   'K':None, 'Ca':None, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':None, 'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                  'Rb':None, 'Sr':None,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                  'Cs':None, 'Ba':None, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }
                     
    # Scale factor is used for determining the bonding threshold. 1.2 is a heuristic that give some lattitude in defining bonds since the UFF radii correspond to equilibrium lengths. 
    scale_factor = 1.2

    # Print warning for uncoded elements.
    for i in Elements:
        if i not in Radii.keys():
            print("ERROR in Table_generator: The geometry contains an element ({}) that the Table_generator function doesn't have bonding information for. This needs to be directly added to the Radii".format(i)+\
                  " dictionary before proceeding. Exiting...")
            quit()

    # Generate distance matrix holding atom-atom separations (only save upper right)
    Dist_Mat = np.triu(cdist(Geometry,Geometry))
    
    # Find plausible connections
    x_ind,y_ind = np.where( (Dist_Mat > 0.0) & (Dist_Mat < max([ Radii[i]**2.0 for i in Radii.keys() ])) )

    # Initialize Adjacency Matrix
    Adj_mat = np.zeros([len(Geometry),len(Geometry)])

    # Iterate over plausible connections and determine actual connections
    for count,i in enumerate(x_ind):
        
        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if Dist_Mat[i,y_ind[count]] < (Radii[Elements[i]]+Radii[Elements[y_ind[count]]])*scale_factor:            
            Adj_mat[i,y_ind[count]]=1
    
    # Hermitize Adj_mat
    Adj_mat=Adj_mat + Adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = { i:0 for i in Radii.keys() }
    conditions = { "H":1, "C":4, "F":1, "Cl":1, "Br":1, "I":1, "O":2, "N":4, "B":4 }
    for count_i,i in enumerate(Adj_mat):

        if Max_Bonds[Elements[count_i]] is not None and sum(i) > Max_Bonds[Elements[count_i]]:
            problem_dict[Elements[count_i]] += 1
            cons = sorted([ (Dist_Mat[count_i,count_j],count_j) if count_j > count_i else (Dist_Mat[count_j,count_i],count_j) for count_j,j in enumerate(i) if j == 1 ])[::-1]
            while sum(Adj_mat[count_i]) > Max_Bonds[Elements[count_i]]:
                sep,idx = cons.pop(0)
                Adj_mat[count_i,idx] = 0
                Adj_mat[idx,count_i] = 0

    # Print warning messages for obviously suspicious bonding motifs.
    if sum( [ problem_dict[i] for i in problem_dict.keys() ] ) > 0:
        print("Table Generation Warnings:")
        for i in sorted(problem_dict.keys()):
            if problem_dict[i] > 0:
                if i == "H":  print("WARNING in Table_generator: {} hydrogen(s) have more than one bond.".format(problem_dict[i]))
                if i == "C":  print("WARNING in Table_generator: {} carbon(s) have more than four bonds.".format(problem_dict[i]))
                if i == "Si": print("WARNING in Table_generator: {} silicons(s) have more than four bonds.".format(problem_dict[i]))
                if i == "F":  print("WARNING in Table_generator: {} fluorine(s) have more than one bond.".format(problem_dict[i]))
                if i == "Cl": print("WARNING in Table_generator: {} chlorine(s) have more than one bond.".format(problem_dict[i]))
                if i == "Br": print("WARNING in Table_generator: {} bromine(s) have more than one bond.".format(problem_dict[i]))
                if i == "I":  print("WARNING in Table_generator: {} iodine(s) have more than one bond.".format(problem_dict[i]))
                if i == "O":  print("WARNING in Table_generator: {} oxygen(s) have more than two bonds.".format(problem_dict[i]))
                if i == "N":  print("WARNING in Table_generator: {} nitrogen(s) have more than four bonds.".format(problem_dict[i]))
                if i == "B":  print("WARNING in Table_generator: {} bromine(s) have more than four bonds.".format(problem_dict[i]))
                
        print("")

    return Adj_mat

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

# Description: Simple wrapper function for writing a mol (V2000) file                                                                      
#                                                                                                                                          
# Inputs      name:     string holding the filename of the output                                                                          
#             elements: list of element types (list of strings)                                                                            
#             geo:      Nx3 array holding the cartesian coordinates of the                                                                 
#                       geometry (atoms are indexed to the elements in Elements)                                                           
#             adj_mat:  NxN array holding the molecular graph                                                                              
#                                                                                                                                          
# Returns     None                                                                                                                         
#                                                                                                                                          
def mol_write(name,elements,geo,adj_mat,append_opt=False):

    # Consistency check                                                                                                                    
    if len(elements) >= 1000:
        print("ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return

    # Check for append vs overwrite condition                                                                                              
    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'

    # Parse the atomtypes                                                                                                              
    atom_types = id_types(elements,adj_mat,2)

    # Get the bond orders                                                                                                              
    bond_mat = find_lewis(atom_types,adj_mat,b_mat_only=True)[0]

    # Parse the basename for the mol header                                                                                                
    base_name = name.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    # Write the file
    with open(name,open_cond) as f:

        # Write the header                                                                                                                 
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))
        
        # Write the number of atoms and bonds                                                                                              
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(len(elements),int(np.sum(adj_mat/2.0))))

        # Write the geometry                                                                                                               
        for count_i,i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n".format(geo[count_i][0],geo[count_i][1],geo[count_i][2],i))
            
        # Write the bonds                                                                                                                  
        bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 and count_j > count_i ]
        for i in bonds:

            # Calculate bond order from the bond_mat                                                                                       
            bond_order = int(bond_mat[i[0],i[1]])

            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(i[0]+1,i[1]+1,bond_order))
        f.write("M  END\n$$$$\n")

    return

# identifies the taffi atom types from an adjacency matrix/list (A) and element identify. 
def id_types(elements,A,gens=2,which_ind=None,avoid=[],geo=None,hybridizations=[]):

    # On first call initialize dictionaries
    if not hasattr(id_types, "mass_dict"):

        # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
        # NOTE: It's inefficient to reinitialize this dictionary every time this function is called
        id_types.mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                             'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                              'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                             'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                             'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                             'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                             'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                             'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    # Assemble prerequisite masses and Loop over the inidices that need to be id'ed
    masses = [ id_types.mass_dict[i] for i in elements ]
    atom_types = [ "["+taffi_type(i,elements,A,masses,gens)+"]" for i in range(len(elements)) ]

    # Add ring atom designation for atom types that belong are intrinsic to rings 
    # (depdends on the value of gens)
    for count_i,i in enumerate(atom_types):
        if ring_atom_specify(A,count_i,ring_size=(gens+2)) == True:
            atom_types[count_i] = "R" + atom_types[count_i]            

    return atom_types

# adjacency matrix based algorithm for identifying the taffi atom type
def taffi_type(ind,elements,adj_mat,masses,gens=2,avoid=[]):

    # On first call initialize dictionaries
    if not hasattr(taffi_type, "periodic"):

        # Initialize periodic table
        taffi_type.periodic = { "h": 1,  "he": 2,\
                               "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                               "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                "k":19,  "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                               "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                               "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

    # Find connections, avoid is used to avoid backtracking
    cons = [ count_i for count_i,i in enumerate(adj_mat[ind]) if i == 1 and count_i not in avoid ]

    # Sort the connections based on the hash function 
    if len(cons) > 0:
        cons = list(zip(*sorted([ (atom_hash(i,adj_mat,masses,gens=gens-1),i) for i in cons ])[::-1]))[1]

    # Calculate the subbranches
    # NOTE: recursive call with the avoid list results 
    if gens == 0:
        subs = []
    else:
        subs = [ taffi_type(i,elements,adj_mat,masses,gens=gens-1,avoid=[ind]) for i in cons ]

    return "{}".format(taffi_type.periodic[elements[ind].lower()]) + "".join([ "["+i+"]" for i in subs ])

# Return true if idx is a ring atom
def ring_atom(adj_mat,idx):

    # Find the atoms connected to atom idx
    connections = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 ]

    # If there isn't at least two connections then there is no possibility that this is a ring atom
    if len(connections) < 2:
        return False

    # Loop through all unique pairs of atom indices connected to idx
    for i in combinations(connections,2):

        # If two atoms connected to atom idx have identical sets of connections then idx is a ring atom
        if return_connected(adj_mat,start=i[0],avoid=[idx]) == return_connected(adj_mat,start=i[1],avoid=[idx]):
            return True

    # If the fuction gets to this point then idx is not a ring atom
    return False

# Return true if idx is a ring atom with specified conditions
def ring_atom_specify(adj_mat,idx,start=None,ring_size=10,counter=0,avoid=[]):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    
    # Loop over connections and recursively search for idx
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid ]
    if len(cons) == 0:
        return False
    elif start in cons:
        return True
    else:
        for i in cons:
            if ring_atom_specify(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid=[idx]) == True:
                return True
        return False

# Returns the set of connected nodes to the start node, while avoiding any connections through nodes in the avoid list. 
def return_connected(adj_mat,start=0,avoid=[]):

    # Initialize the avoid list with the starting index
    avoid = set(avoid+[start])

    # new_0 holds the most recently encountered nodes, beginning with start
    # new_1 is a set holding all of the encountered nodes
    new_0 = [start]
    new_1 = set([start])

    # keep looping until no new nodes are encountered
    while len(new_0) > 0:        

        # reinitialize new_0 with new connections
        new_0 = [ count_j for i in new_0 for count_j,j in enumerate(adj_mat[i]) if j == 1 and count_j not in avoid ]

        # update the new_1 set and avoid list with the most recently encountered new nodes
        new_1.update(new_0)
        avoid.update(new_0)

    # return the set of encountered nodes
    return new_1

# hashing function for canonicalizing geometries on the basis of their adjacency matrices and elements
# ind  : index of the atom being hashed
# A    : adjacency matrix
# M    : masses of the atoms in the molecule
# gens : depth of the search used for the hash   
def atom_hash(ind,A,M,alpha=100.0,beta=0.1,gens=10):    
    if gens <= 0:
        return rec_sum(ind,A,M,beta,gens=0)
    else:
        return alpha * sum(A[ind]) + rec_sum(ind,A,M,beta,gens)

# recursive function for summing up the masses at each generation of connections. 
def rec_sum(ind,A,M,beta,gens,avoid_list=[]):
    if gens != 0:
        tmp = M[ind]*beta
        new = [ count_j for count_j,j in enumerate(A[ind]) if j == 1 and count_j not in avoid_list ]
        if len(new) > 0:
            for i in new:
                tmp += rec_sum(i,A,M,beta*0.1,gens-1,avoid_list=avoid_list+[ind])
            return tmp
        else:
            return tmp
    else:
        return M[ind]*beta

# Description:
# Rotate Point by an angle, theta, about the vector with an orientation of v1 passing through v2. 
# Performs counter-clockwise rotations (i.e., if the direction vector were pointing
# at the spectator, the rotations would appear counter-clockwise)
# For example, a 90 degree rotation of a 0,0,1 about the canonical 
# y-axis results in 1,0,0.
#
# Point: 1x3 array, coordinates to be rotated
# v1: 1x3 array, point the rotation passes through
# v2: 1x3 array, rotation direction vector
# theta: scalar, magnitude of the rotation (defined by default in degrees)
def axis_rot(Point,v1,v2,theta,mode='angle'):

    # Temporary variable for performing the transformation
    rotated=np.array([Point[0],Point[1],Point[2]])

    # If mode is set to 'angle' then theta needs to be converted to radians to be compatible with the
    # definition of the rotation vectors
    if mode == 'angle':
        theta = theta*np.pi/180.0

    # Rotation carried out using formulae defined here (11/22/13) http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/)
    # Adapted for the assumption that v1 is the direction vector and v2 is a point that v1 passes through
    a = v2[0]
    b = v2[1]
    c = v2[2]
    u = v1[0]
    v = v1[1]
    w = v1[2]
    L = u**2 + v**2 + w**2

    # Rotate Point
    x=rotated[0]
    y=rotated[1]
    z=rotated[2]

    # x-transformation
    rotated[0] = ( a * ( v**2 + w**2 ) - u*(b*v + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*x*np.cos(theta) + L**(0.5)*( -c*v + b*w - w*y + v*z )*np.sin(theta)

    # y-transformation
    rotated[1] = ( b * ( u**2 + w**2 ) - v*(a*u + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*y*np.cos(theta) + L**(0.5)*(  c*u - a*w + w*x - u*z )*np.sin(theta)

    # z-transformation
    rotated[2] = ( c * ( u**2 + v**2 ) - w*(a*u + b*v - u*x - v*y - w*z) )\
             * ( 1.0 - np.cos(theta) ) + L*z*np.cos(theta) + L**(0.5)*( -b*u + a*v - v*x + u*y )*np.sin(theta)

    rotated = rotated/L
    return rotated

# Description: This function calls obminimize (open babel geometry optimizer function) to optimize the current geometry
#
# Inputs:      geo:      Nx3 array of atomic coordinates
#              adj_mat:  NxN array of connections
#              elements: N list of element labels
#              ff:       force-field specification passed to obminimize (uff, gaff)
#               q:       total charge on the molecule   
#
# Returns:     geo:      Nx3 array of optimized atomic coordinates
# 
def opt_geo(geo,adj_mat,elements,q=0,ff='uff',step=100):

    # Write a temporary molfile for obminimize to use
    tmp_filename = '.tmp.mol'
    count = 0
    while os.path.isfile(tmp_filename):
        count += 1
        if count == 10:
            print("ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_filename = ".tmp" + tmp_filename            
    
    # Use the mol_write function imported from the write_functions.py 
    # to write the current geometry and topology to file
    mol_write(tmp_filename,elements,geo,adj_mat,append_opt=False)
    
    substring = 'obabel {} -O result.xyz --sd --minimize --steps {} --ff {}'.format(tmp_filename,step,ff)
    output = Popen(substring, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE,bufsize=-1).communicate()[0]
    element,geo = xyz_parse("result.xyz") 
    # Remove the tmp file that was read by obminimize
    try:
        os.remove(tmp_filename)
        os.remove("result.xyz")
    except:
        pass
    return geo

# Return bool depending on if the atom is a nitro nitrogen atom
def is_nitro(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1 ] 
    if len(O_ind) >= 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfoxide sulfur atom
def is_sulfoxide(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j])==1] 
    if len(O_ind) == 1:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_sulfonyl(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j])==1]

    if len(O_ind) == 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a phosphate phosphorus atom
def is_phosphate(i,adj_mat,elements):

    status = False
    if elements[i] not in ["P","p"]:
        return False
    O_ind      = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] ] 
    O_ind_term = [ j for j in O_ind if sum(adj_mat[j]) == 1 ]
    if len(O_ind) == 4 and sum(adj_mat[i]) == 4 and len(O_ind_term) > 0:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_cyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 2 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

# Return bool depending on if the atom is a cyano nitrogen atom
def is_isocyano(i,adj_mat,elements):

    status = False
    if elements[i] not in ["N","n"] or sum(adj_mat[i]) > 1:
        return False
    C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["c","C"] and sum(adj_mat[count_j]) == 1 ]
    if len(C_ind) == 1:
        return True
    else:
        return False

# is function for fragments
# Return bool depending on if the atom is a sulfoxide sulfur atom
def is_frag_sulfoxide(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1] 
    connect = sum(adj_mat[i])

    if len(O_ind) == 1 and int(connect) >= 2:
        return True
    else:
        return False

# Return bool depending on if the atom is a sulfonyl sulfur atom
def is_frag_sulfonyl(i,adj_mat,elements):

    status = False
    if elements[i] not in ["S","s"]:
        return False
    O_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in ["o","O"] and sum(adj_mat[count_j]) == 1] 
    connect = sum(adj_mat[i])

    if len(O_ind) == 2 and int(connect) >= 3:
        return True
    else:
        return False
    
# Returns an NxN matrix holding the bond orders between all atoms in the molecular structure.
# 
# Inputs:  elements:  a list of element labels indexed to the adj_mat
#          adj_mat:   a list of bonds indexed to the elements list
#
# Returns: 
#          bond_mat:  an NxN matrix holding the bond orders between all atoms in the adj_mat
#
def find_lewis(atomtypes,adj_mat_0,q_tot=0,bonding_pref=[],fixed_bonds=[],verbose=False,b_mat_only=False):
    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(find_lewis, "sat_dict"):

        find_lewis.lone_e = {'h':0, 'he':2,\
                             'li':0, 'be':2,                                                                                                                'b':0,     'c':0,     'n':2,     'o':4,     'f':6,    'ne':8,\
                             'na':0, 'mg':2,                                                                                                               'al':0,    'si':0,     'p':2,     's':4,    'cl':6,    'ar':8,\
                             'k':0, 'ca':2, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':None, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':None, 'ge':0,    'as':3,    'se':4,    'br':6,    'kr':None,\
                             'rb':0, 'sr':2,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':None, 'sn':None, 'sb':None, 'te':None,  'i':6,    'xe':None,\
                             'cs':0, 'ba':2, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':None, 'pb':None, 'bi':None, 'po':None, 'at':None, 'rn':None }

        # Initialize periodic table
        find_lewis.periodic = { "h": 1,  "he": 2,\
                                 "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                                 "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                  "k":19, "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                                 "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                                 "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}
        
        # Initialize periodic table
        find_lewis.atomic_to_element = { find_lewis.periodic[i]:i for i in find_lewis.periodic.keys() }
        
        # Electronegativity ordering (for determining lewis structure)
        find_lewis.en = { "h" :2.3,  "he":4.16,\
                          "li":0.91, "be":1.58,                                                                                                               "b" :2.05, "c" :2.54, "n" :3.07, "o" :3.61, "f" :4.19, "ne":4.79,\
                          "na":0.87, "mg":1.29,                                                                                                               "al":1.61, "si":1.91, "p" :2.25, "s" :2.59, "cl":2.87, "ar":3.24,\
                          "k" :0.73, "ca":1.03, "sc":1.19, "ti":1.38, "v": 1.53, "cr":1.65, "mn":1.75, "fe":1.80, "co":1.84, "ni":1.88, "cu":1.85, "zn":1.59, "ga":1.76, "ge":1.99, "as":2.21, "se":2.42, "br":2.69, "kr":2.97,\
                          "rb":0.71, "sr":0.96, "y" :1.12, "zr":1.32, "nb":1.41, "mo":1.47, "tc":1.51, "ru":1.54, "rh":1.56, "pd":1.58, "ag":1.87, "cd":1.52, "in":1.66, "sn":1.82, "sb":1.98, "te":2.16, "i" :2.36, "xe":2.58,\
                          "cs":0.66, "ba":0.88, "la":1.09, "hf":1.16, "ta":1.34, "w" :1.47, "re":1.60, "os":1.65, "ir":1.68, "pt":1.72, "au":1.92, "hg":1.76, "tl":1.79, "pb":1.85, "bi":2.01, "po":2.19, "at":2.39, "rn":2.60} 

    # Initalize elementa and atomic_number lists for use by the function
    elements = [ find_lewis.atomic_to_element[int(i.split("[")[1].split("]")[0])] for i in atomtypes ]
    atomic_number = [ int(i.split("[")[1].split("]")[0]) for i in atomtypes ]
    adj_mat = deepcopy(adj_mat_0)

    # Initially assign all valence electrons as lone electrons
    lone_electrons    = np.zeros(len(atomtypes),dtype="int")    
    bonding_electrons = np.zeros(len(atomtypes),dtype="int")    
    core_electrons    = np.zeros(len(atomtypes),dtype="int")
    valence           = np.zeros(len(atomtypes),dtype="int")
    bonding_target    = np.zeros(len(atomtypes),dtype="int")
    valence_list      = np.zeros(len(atomtypes),dtype="int")    
    
    for count_i,i in enumerate(atomtypes):

        # Grab the total number of (expected) electrons from the atomic number
        N_tot = int(i.split('[')[1].split(']')[0])

        # Determine the number of core/valence electrons based on row in the periodic table
        if N_tot > 54:
            print("ERROR in find_lewis: the algorithm isn't compatible with atomic numbers greater than 54 owing to a lack of rules for treating lanthanides. Exiting...")
            quit()
        elif N_tot > 36:
            N_tot -= 36
            core_electrons[count_i] = 36
            valence[count_i]        = 18
        elif N_tot > 18:
            N_tot -= 18
            core_electrons[count_i] = 18
            valence[count_i]        = 18
        elif N_tot > 10:
            N_tot -= 10
            core_electrons[count_i] = 10
            valence[count_i]        = 8
        elif N_tot > 2:
            N_tot -= 2
            core_electrons[count_i] = 2
            valence[count_i]        = 8
        lone_electrons[count_i] = N_tot
        valence_list[count_i] = N_tot

        # Assign target number of bonds for this atom
        if count_i in [ j[0] for j in bonding_pref ]:
            bonding_target[count_i] = next( j[1] for j in bonding_pref if j[0] == count_i )
        else:
            bonding_target[count_i] = N_tot - find_lewis.lone_e[find_lewis.atomic_to_element[int(i.split('[')[1].split(']')[0])]]

    # Loop over the adjmat and assign initial bonded electrons assuming single bonds (and adjust lone electrons accordingly)
    for count_i,i in enumerate(adj_mat_0):
        bonding_electrons[count_i] += sum(i)
        lone_electrons[count_i] -= sum(i)
    
    # Eliminate all radicals by forming higher order bonds
    change_list = range(len(lone_electrons))
    bonds_made = []    
    loop_list   = [ (atomic_number[i],i) for i in range(len(lone_electrons)) ]
    loop_list   = [ i[1] for i in sorted(loop_list) ]

    # Check for special chemical groups
    for i in range(len(atomtypes)):

        # Handle nitro groups
        if is_nitro(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ]
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],1)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            lone_electrons[O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[1],i] += 1

        # Handle sulfoxide groups
        if is_sulfoxide(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the thioketone atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind[0])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[O_ind[0],i] += 1

        # Handle sulfonyl groups
        if is_sulfonyl(i,adj_mat_0,elements) is True:            
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the sulfoxide atoms from the bonding_pref list
            bonding_pref += [(i,6)]
            bonding_pref += [(O_ind[0],2)]
            bonding_pref += [(O_ind[1],2)]
            bonding_electrons[O_ind[0]] += 1
            bonding_electrons[O_ind[1]] += 1
            bonding_electrons[i] += 2
            lone_electrons[O_ind[0]] -= 1
            lone_electrons[O_ind[1]] -= 1
            lone_electrons[i] -= 2
            bonds_made += [(i,O_ind[0])]
            bonds_made += [(i,O_ind[1])]
            adj_mat[i,O_ind[0]] += 1
            adj_mat[i,O_ind[1]] += 1
            adj_mat[O_ind[0],i] += 1
            adj_mat[O_ind[1],i] += 1            
        
        # Handle phosphate groups 
        if is_phosphate(i,adj_mat_0,elements) is True:
            O_ind      = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] in ["o","O"] ] # Index of single bonded O-P oxygens
            O_ind_term = [ j for j in O_ind if sum(adj_mat_0[j]) == 1 ] # Index of double bonded O-P oxygens
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the phosphate atoms from the bonding_pref list
            bonding_pref += [(i,5)]
            bonding_pref += [(O_ind_term[0],2)]  # during testing it ended up being important to only add a bonding_pref tuple for one of the terminal oxygens
            bonding_electrons[O_ind_term[0]] += 1
            bonding_electrons[i] += 1
            lone_electrons[O_ind_term[0]] -= 1
            lone_electrons[i] -= 1
            bonds_made += [(i,O_ind_term[0])]
            adj_mat[i,O_ind_term[0]] += 1
            adj_mat[O_ind_term[0],i] += 1

        # Handle cyano groups
        if is_cyano(i,adj_mat_0,elements) is True:
            C_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j] in  ["c","C"] and sum(adj_mat_0[count_j]) == 2 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in C_ind ] # remove bonds involving the cyano atoms from the bonding_pref list
            bonding_pref += [(i,3)]
            bonding_pref += [(C_ind[0],4)]
            bonding_electrons[C_ind[0]] += 2
            bonding_electrons[i] += 2
            lone_electrons[C_ind[0]] -= 2
            lone_electrons[i] -= 2
            bonds_made += [(i,C_ind[0])]
            bonds_made += [(i,C_ind[0])]
            adj_mat[i,C_ind[0]] += 2
            adj_mat[C_ind[0],i] += 2

        # Handle isocyano groups
        if is_isocyano(i,adj_mat,elements) is True:
            C_ind = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and elements[count_j] in  ["c","C"] and sum(adj_mat[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in C_ind ] # remove bonds involving the cyano atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(C_ind[0],3)]
            bonding_electrons[C_ind[0]] += 2
            bonding_electrons[i] += 2
            lone_electrons[C_ind[0]] -= 2
            lone_electrons[i] -= 2
            bonds_made += [(i,C_ind[0])]
            bonds_made += [(i,C_ind[0])]
            adj_mat[i,C_ind[0]] += 2
            adj_mat[C_ind[0],i] += 2

    # Apply fixed_bonds argument
    off_limits=[]
    for i in fixed_bonds:

        # Initalize intermediate variables
        a = i[0]
        b = i[1]
        N = i[2]
        N_current = len([ j for j in bonds_made if (a,b) == j or (b,a) == j ]) + 1
        # Check that a bond exists between these atoms in the adjacency matrix
        if adj_mat_0[a,b] != 1:
            print("ERROR in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but the adjacency matrix doesn't reflect a bond. Exiting...")
            quit()

        # Check that less than or an equal number of bonds exist between these atoms than is requested
        if N_current > N:
            print("ERROR in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but {} bonds already exist between these atoms. There may be a conflict".format(N_current))
            print("                      between the special groups handling and the requested lewis_structure.")
            quit()

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[a] < (N - N_current):
            print("Warning in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(atomtypes[a],lone_electrons[a]))

        # Check that enough lone electrons exists on each atom to reach the target bond number
        if lone_electrons[b] < (N - N_current):
            print("Warning in find_lewis: fixed_bonds requests bond creation between atoms {} and {} ({} bonds)".format(a,b,N))
            print("                      but atom {} only has {} lone electrons.".format(atomtypes[b],lone_electrons[b]))
        

        # Make the bonds between the atoms
        for j in range(N-N_current):
            bonding_electrons[a] += 1
            bonding_electrons[b] += 1
            lone_electrons[a]    -= 1
            lone_electrons[b]    -= 1
            bonds_made += [ (a,b) ]

        # Append bond to off_limits group so that further bond additions/breaks do not occur.
        off_limits += [(a,b),(b,a)]

    # Turn the off_limits list into a set for rapid lookup
    off_limits = set(off_limits)

    # Add/remove electrons depending on the total charge of the molecule
    happy = [ i[0] for i in bonding_pref if i[1] <= bonding_electrons[i[0]]]
    adjust_ind = [ count_i for count_i,i in enumerate(lone_electrons) if i > 0 and count_i not in happy ]

    # Determine is electrons need to be removed or added
    if q_tot > 0: adjust = -1
    else: adjust = 1

    # Adjust the number of electrons by removing or adding to the available lone pairs
    # The algorithm simply adds/removes from the first N lone pairs that are discovered
    if len(adjust_ind) >= abs(q_tot): 
        for i in range(abs(q_tot)): lone_electrons[adjust_ind[i]] += adjust
    else:
        for i in range(abs(q_tot)): lone_electrons[0] += adjust

    # Initialize objects for use in the algorithm
    lewis_total = 100
    lewis_lone_electrons = []
    lewis_bonding_electrons = []
    lewis_core_electrons = []
    lewis_valence = []
    lewis_bonding_target = []
    lewis_bonds_made = []
    lewis_adj_mat = []
    lewis_bonds_en = []
    
    # The outer loop checks each bonding structure produced by the inner loop for consistency with
    # the user specified "pref_bonding" and pref_argument with bonding electrons are
    for dummy_counter in range(lewis_total):
        lewis_loop_list = loop_list
        random.shuffle(lewis_loop_list)
        outer_counter     = 0
        inner_max_cycles  = 100
        outer_max_cycles  = 100
        bond_sat = False
        
        lewis_lone_electrons.append(deepcopy(lone_electrons))
        lewis_bonding_electrons.append(deepcopy(bonding_electrons))
        lewis_core_electrons.append(deepcopy(core_electrons))
        lewis_valence.append(deepcopy(valence))
        lewis_bonding_target.append(deepcopy(bonding_target))
        lewis_bonds_made.append(deepcopy(bonds_made))
        lewis_adj_mat.append(deepcopy(adj_mat))
        lewis_counter = len(lewis_lone_electrons) - 1
        lewis_bonds_en.append(0)
        # Adjust the number of electrons by removing or adding to the available lone pairs
        # The algorithm simply adds/removes from the first N lone pairs that are discovered

        random.shuffle(adjust_ind)
        if len(adjust_ind) >= abs(q_tot): 
            for i in range(abs(q_tot)): lewis_lone_electrons[-1][adjust_ind[i]] += adjust
        else:
            for i in range(abs(q_tot)): lewis_lone_electrons[-1][0] += adjust

        # Search for an optimal lewis structure
        while bond_sat is False:
        
            # Initialize necessary objects
            change_list   = range(len(lewis_lone_electrons[lewis_counter]))
            inner_counter = 0
            bond_sat = True                
            # Inner loop forms bonds to remove radicals or underbonded atoms until no further
            # changes in the bonding pattern are observed.
            while len(change_list) > 0:
                change_list = []
                for i in lewis_loop_list:

                    # List of atoms that already have a satisfactory binding configuration.
                    happy = [ j[0] for j in bonding_pref if j[1] <= lewis_bonding_electrons[lewis_counter][j[0]]]            
                    
                    # If the current atom already has its target configuration then no further action is taken
                    if i in happy: continue

                    # If there are no lone electrons or too more bond formed then skip
                    if lewis_lone_electrons[lewis_counter][i] == 0: continue
                    
                    # Take action if this atom has a radical or an unsatifisied bonding condition
                    if lewis_lone_electrons[lewis_counter][i] % 2 != 0 or lewis_bonding_electrons[lewis_counter][i] != lewis_bonding_target[lewis_counter][i]:
 
                        # Try to form a bond with a neighboring radical (valence +1/-1 check ensures that no improper 5-bonded atoms are formed)
                        lewis_bonded_radicals = [ (-find_lewis.en[elements[count_j]],count_j) for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and lewis_lone_electrons[lewis_counter][count_j] % 2 != 0 \
                                           and 2*(lewis_bonding_electrons[lewis_counter][count_j]+1)+(lewis_lone_electrons[lewis_counter][count_j]-1) <= lewis_valence[lewis_counter][count_j] and lewis_lone_electrons[lewis_counter][count_j]-1 >= 0 and count_j not in happy ]
                        lewis_bonded_lonepairs= [ (-find_lewis.en[elements[count_j]],count_j) for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and lewis_lone_electrons[lewis_counter][count_j] > 0 \
                                           and 2*(lewis_bonding_electrons[lewis_counter][count_j]+1)+(lewis_lone_electrons[lewis_counter][count_j]-1) <= lewis_valence[lewis_counter][count_j] and lewis_lone_electrons[lewis_counter][count_j]-1 >= 0 and count_j not in happy ]
                        
                        # Sort by atomic number (cheap way of sorting carbon before other atoms, should probably switch over to electronegativities) 
                        lewis_bonded_radicals = [ j[1] for j in  sorted(lewis_bonded_radicals) ]
                        lewis_bonded_lonepairs = [ j[1] for j in  sorted(lewis_bonded_lonepairs) ]

                        # Correcting radicals is attempted first
                        if len(lewis_bonded_radicals) > 0:
                            lewis_bonding_electrons[lewis_counter][i] += 1
                            lewis_bonding_electrons[lewis_counter][lewis_bonded_radicals[0]] += 1
                            lewis_adj_mat[lewis_counter][i][lewis_bonded_radicals[0]] += 1
                            lewis_adj_mat[lewis_counter][lewis_bonded_radicals[0]][i] += 1 
                            lewis_bonds_en[lewis_counter] += 1.0/find_lewis.en[elements[i]]/find_lewis.en[elements[lewis_bonded_radicals[0]]]
                            lewis_lone_electrons[lewis_counter][i] -= 1
                            lewis_lone_electrons[lewis_counter][lewis_bonded_radicals[0]] -= 1
                            change_list += [i,lewis_bonded_radicals[0]]
                            lewis_bonds_made[lewis_counter] += [(i,lewis_bonded_radicals[0])]

                        # Else try to form a bond with a neighboring atom with spare lone electrons (valence check ensures that no improper 5-bonded atoms are formed)
                        elif len(lewis_bonded_lonepairs) > 0:
                            lewis_bonding_electrons[lewis_counter][i] += 1
                            lewis_bonding_electrons[lewis_counter][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][i][lewis_bonded_lonepairs[0]] += 1
                            lewis_adj_mat[lewis_counter][lewis_bonded_lonepairs[0]][i] += 1
                            lewis_bonds_en[lewis_counter] += 1.0/find_lewis.en[elements[i]]/find_lewis.en[elements[lewis_bonded_lonepairs[0]]]
                            lewis_lone_electrons[lewis_counter][i] -= 1
                            lewis_lone_electrons[lewis_counter][lewis_bonded_lonepairs[0]] -= 1
                            change_list += [i,lewis_bonded_lonepairs[0]]
                            lewis_bonds_made[lewis_counter] += [(i,lewis_bonded_lonepairs[0])]

                # Increment the counter and break if the maximum number of attempts have been made
                inner_counter += 1
                if inner_counter >= inner_max_cycles:
                    print("WARNING: maximum attempts to establish a reasonable lewis-structure exceeded ({}).".format(inner_max_cycles))

            # Check if the user specified preferred bond order has been achieved.
            if bonding_pref is not None:
                unhappy = [ i[0] for i in bonding_pref if i[1] != lewis_bonding_electrons[lewis_counter][i[0]]]            
                if len(unhappy) > 0:

                    # Break the first bond involving one of the atoms bonded to the under/over coordinated atoms
                    ind = set([unhappy[0]] + [ count_i for count_i,i in enumerate(adj_mat_0[unhappy[0]]) if i == 1 and (count_i,unhappy[0]) not in off_limits ])

                    # Check if a rearrangment is possible, break if none are available
                    try:
                        break_bond = next( i for i in lewis_bonds_made[lewis_counter] if i[0] in ind or i[1] in ind )
                    except:
                        print("WARNING: no further bond rearrangments are possible and bonding_pref is still not satisfied.")
                        break
                    
                    # Perform bond rearrangment
                    lewis_bonding_electrons[lewis_counter][break_bond[0]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[0]] += 1
                    lewis_adj_mat[lewis_counter][break_bond[0]][break_bond[1]] -= 1
                    lewis_adj_mat[lewis_counter][break_bond[1]][break_bond[0]] -= 1
                    lewis_bonding_electrons[lewis_counter][break_bond[1]] -= 1
                    lewis_lone_electrons[lewis_counter][break_bond[1]] += 1

                    # Remove the bond from the list and reorder lewis_loop_list so that the indices involved in the bond are put last                
                    lewis_bonds_made[lewis_counter].remove(break_bond)
                    lewis_loop_list.remove(break_bond[0])
                    lewis_loop_list.remove(break_bond[1])
                    lewis_loop_list += [break_bond[0],break_bond[1]]
                    # Update the bond_sat flag
                    bond_sat = False

                    # Increment the counter and break if the maximum number of attempts have been made
                    outer_counter += 1
                    
                    # Periodically reorder the list to avoid some cyclical walks
                    if outer_counter % 100 == 0:
                        lewis_loop_list = reorder_list(lewis_loop_list,atomic_number)

                    # Print diagnostic upon failure
                    if outer_counter >= outer_max_cycles:
                        print("WARNING: maximum attempts to establish a lewis-structure consistent")
                        print("         with the user supplied bonding preference has been exceeded ({}).".format(outer_max_cycles))
                        break
            
        # Delete last entry in the lewis arrays if the electronic structure is not unique
        if array_unique(lewis_adj_mat[-1],lewis_adj_mat[:-1]) is False:
            lewis_lone_electrons    = lewis_lone_electrons[:-1]
            lewis_bonding_electrons = lewis_bonding_electrons[:-1]
            lewis_core_electrons    = lewis_core_electrons[:-1]
            lewis_valence           = lewis_valence[:-1]
            lewis_bonding_target    = lewis_bonding_target[:-1]
            lewis_bonds_made        = lewis_bonds_made[:-1]
            lewis_adj_mat           = lewis_adj_mat[:-1]
            lewis_bonds_en          = lewis_bonds_en[:-1]

    # Find the total number of lone electrons in each structure
    lone_electrons_sums = []
    for i in range(len(lewis_lone_electrons)):
        lone_electrons_sums.append(sum(lewis_lone_electrons[i]))
    
    # Find the total formal charge for each structure
    formal_charges_sums = []
    for i in range(len(lewis_lone_electrons)):
        fc = 0
        for j in range(len(atomtypes)):
            fc += valence_list[j] - lewis_bonding_electrons[i][j] - lewis_lone_electrons[i][j]
        formal_charges_sums.append(fc)

    # Add the total number of radicals to the total formal charge to determine the criteria.
    # The radical count is scaled by 0.01 and the lone pair count is scaled by 0.001. This results
    # in the structure with the lowest formal charge always being returned, and the radical count 
    # only being considered if structures with equivalent formal charges are found, and likewise with
    # the lone pair count. The structure(s) with the lowest score will be returned.
    lewis_criteria = []
    for i in range(len(lewis_lone_electrons)):
        lewis_criteria.append( abs(formal_charges_sums[i]) + 0.1*sum([ 1 for j in lewis_lone_electrons[i] if j % 2 != 0 ]) + 0.01*lewis_bonds_en[i])

    best_lewis = [i[0] for i in sorted(enumerate(lewis_criteria), key=lambda x:x[1])]  # sort from least to most and return a list containing the origial list's indices in the correct order
    best_lewis = [ i for i in best_lewis if lewis_criteria[i] == lewis_criteria[best_lewis[0]] ]

    # If only the bonding matrix is requested, then only that is returned
    if b_mat_only is True:
        return [ lewis_adj_mat[_] for _ in best_lewis ]

    # Optional bonding pref return to handle cases with special groups
    return [ lewis_lone_electrons[_] for _ in best_lewis ],[ lewis_bonding_electrons[_] for _ in best_lewis ],\
        [ lewis_core_electrons[_] for _ in best_lewis ],[ lewis_adj_mat[_] for _ in best_lewis ],[ formal_charges_sums[_] for _ in best_lewis ]


# Description: Checks if an array "a" is unique compared with a list of arrays "a_list"
#              at the first match False is returned.
def array_unique(a,a_list):
    for i in a_list:
        if np.array_equal(a,i):
            return False
    return True

# Returns a matrix of graphical separations for all nodes in a graph defined by the inputted adjacency matrix 
def graph_seps(adj_mat_0):

    # Create a new name for the object holding A**(N), initialized with A**(1)
    adj_mat = deepcopy(adj_mat_0)
    
    # Initialize an np.array to hold the graphical separations with -1 for all unassigned elements and 0 for the diagonal.
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

