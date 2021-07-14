import sys,argparse,os,time,math,subprocess,fnmatch,shutil
from copy import deepcopy
import numpy as np
from math import sqrt,sin,acos,cos,tan,factorial
from scipy import cross

# import taffi related functions
from taffi_functions import * 

def main(argv):
    gens=2
    '''test on single ring-contained molecule.'''
    # load in geometry, elements and charge
    E,G =xyz_parse("/XXX/XXX.xyz")
    ring=get_rings(E,G,gens=2,keep_flag=True)
    canon_ring_geo(ring)
    Energy_list=gen_ring_conf(ring,outname="test_conf.xyz")

    return

# This fuction is used to get ring structures from a ring-conatined molecule
# Current version can't deal with fused ring system and only take the first bond matrix
# If return_R0 is True, both RC2 and RC0 will be returned
#
def get_rings(E,G,gens=2,return_R0=True):

    # Define mass dictionary for args.avoid_frags == True case
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    #calculate the adjacency matrix
    adj_mat    = Table_generator(E,G)
    atom_types = id_types(E,adj_mat,2)

    # calculate formal charge and bonding matrix
    bond_mat = find_lewis(E,adj_mat,b_mat_only=True)[0]

    # identify ring atoms and number of rings
    rings=[]
    ring_size_list=range(11)[3:] # at most 10 ring structure
    
    for ring_size in ring_size_list:
        for j,Ej in enumerate(E):
            is_ring,ring_ind = ring_atom(adj_mat,j,ring_size=ring_size)
            if is_ring and ring_ind not in rings:
                rings += [ring_ind]
    rings=[list(ring_inds) for ring_inds in rings]

    # remove large rings (contains more than two rings)
    remove_list=[]
    for ring_inds in rings:
        other_rings=[j for j in rings if len(j) < len(ring_inds)]
        other_ring_inds=set(sum(other_rings,[]))
        new_inds=[j for j in ring_inds if j not in other_ring_inds]
        if len(new_inds)==0: remove_list+=[ring_inds]
    
    for ring_inds in remove_list: rings.remove(ring_inds)
    
    # Initialize ring dictionary    
    rings_depth0={}
    rings_depth1={}
    rings_depth2={}
    lr_0=0
    lr_1=0
    lr_2=0

    for ring_inds in rings:
        # get the nearest side chain atoms (ns, depth=1) and next-nearest ..(nns, depth=2)
        ring_ns    = list(set([j for j,_ in enumerate(E) if sum(adj_mat[ring_inds,:][:,j])>=1 and j not in ring_inds]))
        ring_nns = [j for j,Ej in enumerate(E) if sum(adj_mat[ring_ns,:][:,j])>=1 and j not in ring_inds and Ej != 'H'] # and j not in rings_ns
        ring_ns_nonH = [Ei for Ei in ring_ns if E[Ei] != 'H']

        # ring atoms and side atoms
        ring_ind_depth0 = ring_inds
        ring_ind_depth2 = ring_inds + ring_ns + ring_nns
        heavy_atoms_depth2 = [ind for ind in ring_ind_depth2 if E[ind] is not 'H']

        # first generate depth = 0 RC
        N_adj_mat         =adj_mat[ring_ind_depth0,:][:,ring_ind_depth0]
        N_bond_mat        =bond_mat[ring_ind_depth0,:][:,ring_ind_depth0]
        N_atom_types      =[atom_types[i] for i in ring_ind_depth0]
        N_E               =[E[i] for i in ring_ind_depth0]
        N_G               =G[ring_ind_depth0,:]

        # Include the atoms in the mode and connected atoms within the preserve list
        loop_list  =[ring_ind_depth0.index(j) for j in ring_inds]
        fixed_bonds=[]
        for i in loop_list:
            fixed_bonds += [(i,j,int(k)) for j,k in enumerate(N_bond_mat[i]) if k > 1 and i < j]

        N_Geometry,tmp_Atom_types,N_Elements,N_Adj_mat,added_idx = ring_add_hydrogens(N_G,N_adj_mat,N_bond_mat,deepcopy(N_atom_types),N_E)
        ring_masses     = [ mass_dict[N_Elements[j]] for j in range(len(tmp_Atom_types)) ]
        hash_list,atoms = [ list(j) for j in zip(*sorted([ (atom_hash(count_k,N_Adj_mat,ring_masses,gens=len(ring_inds)),k) for count_k,k in enumerate(range(len(tmp_Atom_types))) ],reverse=True)) ] 
        hash_sum=len(ring_inds)*1e5 + sum(hash_list)
        
        Opt_geo=opt_geo(N_Geometry,N_Adj_mat,N_Elements,ff='mmff94')
        N_ring_inds = [ring_ind_depth0.index(j) for j in ring_inds]
        N_G,N_E,N_atomtypes,N_adj_mat = canon_ring_geo(N_Elements,Opt_geo,tmp_Atom_types,N_Adj_mat,N_ring_inds,atoms)
        
        rings_depth0[lr_0]={}
        rings_depth0[lr_0]["geometry"]   = N_G
        rings_depth0[lr_0]["elements"]   = N_E
        rings_depth0[lr_0]["atomtypes"]  = N_atomtypes
        rings_depth0[lr_0]["adj_mat"]    = N_adj_mat
        rings_depth0[lr_0]["ring_inds"]  = N_ring_inds
        rings_depth0[lr_0]["hash_index"] = "{:< 12.6f}".format(hash_sum)
        rings_depth0[lr_0]["N_heavy"]    = len(ring_ind_depth0)
        rings_depth0[lr_0]["smiles"]     = return_smi(N_E,N_G,N_adj_mat)
        lr_0 += 1    

        # Then generate depth=2 RC
        if len(ring_ns) > 0:
            N_adj_mat         =adj_mat[ring_ind_depth2,:][:,ring_ind_depth2]
            N_bond_mat        =bond_mat[ring_ind_depth2,:][:,ring_ind_depth2]
            N_atom_types      =[atom_types[i] for i in ring_ind_depth2]
            N_E               =[E[i] for i in ring_ind_depth2]
            N_G               =G[ring_ind_depth2,:]

            # Include the atoms in the mode and connected atoms within the preserve list
            loop_list  =[ring_ind_depth2.index(j) for j in ring_ind_depth2]
            fixed_bonds=[]
            for i in loop_list:
                fixed_bonds += [(i,j,int(k)) for j,k in enumerate(N_bond_mat[i]) if k > 1 and i < j]

            N_Geometry,tmp_Atom_types,N_Elements,N_Adj_mat,added_idx = ring_add_hydrogens(N_G,N_adj_mat,N_bond_mat,deepcopy(N_atom_types),N_E)
            ring_masses     = [ mass_dict[N_Elements[j]] for j in range(len(tmp_Atom_types)) ]
            hash_list,atoms = [ list(j) for j in zip(*sorted([ (atom_hash(count_k,N_Adj_mat,ring_masses,gens=len(ring_inds)),k) for count_k,k \
                                                               in enumerate(range(len(tmp_Atom_types))) ],reverse=True)) ] 
            hash_sum=len(ring_inds)*1e5 + sum(hash_list)

            Opt_geo=opt_geo(N_Geometry,N_Adj_mat,N_Elements,ff='mmff94')
            N_ring_inds = [ring_ind_depth0.index(j) for j in ring_inds]
            N_G,N_E,N_atomtypes,N_adj_mat = canon_ring_geo(N_Elements,Opt_geo,tmp_Atom_types,N_Adj_mat,N_ring_inds,atoms)

            rings_depth2[lr_2]={}
            rings_depth2[lr_2]["geometry"]   = N_G
            rings_depth2[lr_2]["elements"]   = N_E
            rings_depth2[lr_2]["atomtypes"]  = N_atomtypes
            rings_depth2[lr_2]["adj_mat"]    = N_adj_mat
            rings_depth2[lr_2]["ring_inds"]  = N_ring_inds
            rings_depth2[lr_2]["hash_index"] = "{:< 12.6f}".format(hash_sum)
            rings_depth2[lr_2]["N_heavy"]    = len(ring_ind_depth2)
            rings_depth2[lr_2]["smiles"]     = return_smi(N_E,N_G,N_adj_mat)
            rings_depth2[lr_2]["ringsides"]  = ring_ns_nonH
            lr_2 += 1    

    if return_R0:
        return rings_depth0,rings_depth2
    else:
        return rings_depth2

# This function is used to give a connect sequence of ring atoms
#
def canon_ring_geo(E,G,atomtypes,adj_mat,ring_inds,atoms):
    
    first=[atom for atom in atoms if atom in ring_inds][0]
    seq=[first]
    cons = [ count_i for count_i,i in enumerate(adj_mat[first]) if i == 1 and count_i in ring_inds]
    second=atoms[min([atoms.index(j) for j in cons])]
    seq+=[second]
    
    for k in range(len(ring_inds)-2):
        seq+=[count_i for count_i,i in enumerate(adj_mat[seq[1+k]]) if i == 1 and count_i in ring_inds and count_i not in seq]
    seq+=[ind for ind in atoms if ind not in seq]

    N_G  = G[seq]
    N_E  = [ E[i] for i in seq ]                                                                                                                                                
    N_atomtypes = [ atomtypes[i] for i in seq ] 
    N_adj_mat   = adj_mat[seq][:,seq]

    return N_G,N_E,N_atomtypes,N_adj_mat

# Add hydrogens based upon the supplied atom types. 
# This function is only compatible with TAFFI atom types
# NOTE: Hydrogenation heuristics for geometry assume carbon behavior. This isn't usually a problem when the results are refined with transify, but more specific rules should be implemented in the future
#
def ring_add_hydrogens(geo,adj_mat_0,bond_mat_0,atomtypes,elements,q_tot=0,preserve=[],saturate=True,retype=True):
    
    # Initialize the saturation dictionary the first time this function is called
    if not hasattr(ring_add_hydrogens, "sat_dict"):
        ring_add_hydrogens.sat_dict = {'H':1, 'He':1,\
                                       'Li':1, 'Be':2,                                                                                                                'B':3,     'C':4,     'N':3,     'O':2,     'F':1,    'Ne':1,\
                                       'Na':1, 'Mg':2,                                                                                                               'Al':3,    'Si':4,     'P':3,     'S':2,    'Cl':1,    'Ar':1,\
                                       'K': 1, 'Ca':2, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':None, 'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                                       'Rb':1, 'Sr':2,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                                       'Cs':1, 'Ba':2, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }

        ring_add_hydrogens.lone_e = {'H':0, 'He':2,\
                                    'Li':0, 'Be':2,                                                                                                                'B':0,     'C':0,     'N':2,     'O':4,     'F':6,    'Ne':8,\
                                    'Na':0, 'Mg':2,                                                                                                               'Al':0,    'Si':0,     'P':2,     'S':4,    'Cl':6,    'Ar':8,\
                                    'K': 0, 'Ca':2, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':None, 'Ge':0,    'As':3,    'Se':4,    'Br':6,    'Kr':None,\
                                    'Rb':0, 'Sr':2,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':6,    'Xe':None,\
                                    'Cs':0, 'Ba':2, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }

        # Initialize periodic table
        ring_add_hydrogens.periodic = { "H": 1,  "He": 2,\
                                        "Li":3,  "Be": 4,                                                                                                      "B":5,    "C":6,    "N":7,    "O":8,    "F":9,    "Ne":10,\
                                        "Na":11, "Mg":12,                                                                                                     "Al":13,  "Ai":14,   "P":15,   "S":16,   "Cl":17,  "Ar":18,\
                                        "K" :19, "Ca":20,  "Sc":21,  "Ti":22,  "V":23,  "Cr":24,  "Mn":25,  "Fe":26,  "Co":27,  "Ni":28,  "Cu":29,  "Zn":30,  "Ga":31,  "Ge":32,  "As":33,  "Se":34,   "Br":35,  "Kr":36,\
                                        "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                                        "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}
        
        ring_add_hydrogens.atomic_to_element = { ring_add_hydrogens.periodic[i]:i for i in ring_add_hydrogens.periodic.keys() }

    # Initalize elementa and atomic_number lists for use by the function
    bond_mat= deepcopy(bond_mat_0)
    adj_mat = deepcopy(adj_mat_0)
    atomic_number = [ ring_add_hydrogens.periodic[i] for i in elements ]

    # Initially assign all valence electrons as lone electrons
    lone_electrons    = np.zeros(len(atomtypes),dtype="int")    
    bonding_electrons = np.zeros(len(atomtypes),dtype="int")    
    core_electrons    = np.zeros(len(atomtypes),dtype="int")
    valence           = np.zeros(len(atomtypes),dtype="int")
    bonding_target    = np.zeros(len(atomtypes),dtype="int")
    valence_list      = np.zeros(len(atomtypes),dtype="int")    

    for count_i,i in enumerate(elements):

        # Grab the total number of (expected) electrons from the atomic number
        N_tot = atomic_number[count_i]

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

    # Loop over the adjmat and assign initial bonded electrons assuming single bonds (and adjust lone electrons accordingly)
    for count_i,i in enumerate(bond_mat_0):
        bonding_electrons[count_i] += sum(i)
        lone_electrons[count_i] -= sum(i)

    # Check for special chemical groups
    bonding_pref=[]
    for i in range(len(elements)):

        # Handle nitro groups
        if is_nitro(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ]
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],1)]
            bonding_pref += [(O_ind[1],2)]

        # Handle sulfoxide groups
        if is_frag_sulfoxide(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref  = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the thioketone atoms from the bonding_pref list
            bonding_pref += [(i,4)]
            bonding_pref += [(O_ind[0],2)]

        # Handle sulfonyl groups
        if is_frag_sulfonyl(i,adj_mat_0,elements) is True:
            O_ind = [ count_j for count_j,j in enumerate(adj_mat_0[i]) if j == 1 and elements[count_j].lower() == "o" and sum(adj_mat_0[count_j]) == 1 ]
            bonding_pref = [ j for j in bonding_pref if j[0] != i and j[0] not in O_ind ] # remove bonds involving the sulfoxide atoms from the bonding_pref list
            bonding_pref += [(i,6)]
            bonding_pref += [(O_ind[0],2)]
            bonding_pref += [(O_ind[1],2)]

    # change list to dict
    bond_pref={}
    for bp in bonding_pref:
        bond_pref[bp[0]]=bp[1] 

    # Intermediate scalars
    H_length = 1.1
    N_atoms  = len(geo)
    init_len = len(geo)

    # Loop over the atoms in the geometry
    for count_i,i in enumerate(geo):
        # ID undercoordinated atoms
        if count_i in preserve:
            continue
        elif ring_add_hydrogens.sat_dict[elements[count_i]] is not None:
            if count_i in bond_pref.keys():
                B_expected = bond_pref[count_i]
            else:
                B_expected = ring_add_hydrogens.sat_dict[elements[count_i]]
        else:
            print("ERROR in add_hydrogens: could not determine the number of hydrogens to add to {}. Exiting...".format(elements[count_i]))
            quit()
        B_current  = bonding_electrons[count_i]

        # Determine the number of nuclei that are attached and expected.
        N_current   = sum(adj_mat_0[count_i])
        N_expected = N_current + (B_expected - B_current)

        # Add hydrogens to undercoordinated atoms
        if N_expected > N_current:

            old_inds = [ count_j for count_j,j in enumerate(bond_mat[count_i]) if j >= 1 ]            
            # Protocols for 1 missing hydrogen
            if N_expected - N_current == 1:
                if N_expected == 1:
                    new = i + np.array([H_length,0.0,0.0])
                elif N_expected == 2:
                    new = -1.0 * normalize(geo[old_inds[0]] - i) * H_length + i + np.array([np.random.random(),np.random.random(),np.random.random()])*0.01 #random factor added for non-carbon types to relax during FF-opt
                elif N_expected == 3:
                    new = -1.0 * normalize( normalize(geo[old_inds[0]] - i) + normalize(geo[old_inds[1]] - i) ) * H_length + i
                elif N_expected == 4:
                    new = -1.0 * normalize( normalize(geo[old_inds[0]] - i) + normalize(geo[old_inds[1]] - i) + normalize(geo[old_inds[2]] - i) ) * H_length + i                

                # Update geometry, adj_mat, elements, and atomtypes with one new atoms
                geo = np.vstack([geo,new])
                atomtypes += ["[1[{}]]".format(atomtypes[count_i].split(']')[0].split('[')[1])]
                elements += ["H"]
                tmp = np.zeros([N_atoms+1,N_atoms+1])
                tmp[:N_atoms,:N_atoms] = adj_mat
                tmp[-1,count_i] = 1
                tmp[count_i,-1] = 1
                adj_mat = tmp                
                N_atoms += 1

            # Protocols for 2 missing hydrogens
            # ISSUE, NEW ALGORITHM IS BASED ON BONDED ATOMS NOT BONDED CENTERS
            if N_expected - N_current == 2:
                if N_expected == 2:
                    new_1 = i + np.array([H_length,0.0,0.0])
                    new_2 = i - np.array([H_length,0.0,0.0])
                elif N_expected == 3:
                    rot_vec = normalize(cross( geo[old_inds[0]] - i, np.array([np.random.random(),np.random.random(),np.random.random()]) ))
                    new_1 = normalize(axis_rot(geo[old_inds[0]],rot_vec,i,120.0) - i)*H_length + i
                    new_2 = normalize(axis_rot(geo[old_inds[0]],rot_vec,i,240.0) - i)*H_length + i
                elif N_expected == 4:
                    bisector = normalize(geo[old_inds[0]] - i + geo[old_inds[1]] - i) 
                    new_1    = axis_rot(geo[old_inds[0]],bisector,i,90.0)
                    new_2    = axis_rot(geo[old_inds[1]],bisector,i,90.0) 
                    rot_vec  = normalize(cross(new_1-i,new_2-i))
                    angle    = ( 109.5 - acos(np.dot(normalize(new_1-i),normalize(new_2-i)))*180.0/np.pi ) / 2.0
                    new_1    = axis_rot(new_1,rot_vec,i,-angle)
                    new_2    = axis_rot(new_2,rot_vec,i,angle)
                    new_1    = -1*H_length*normalize(new_1-i) + i
                    new_2    = -1*H_length*normalize(new_2-i) + i
                    
                # Update geometry, adj_mat, elements, and atomtypes with two new atoms
                geo = np.vstack([geo,new_1])
                geo = np.vstack([geo,new_2])
                atomtypes += ["[1[{}]]".format(atomtypes[count_i].split(']')[0].split('[')[1])]*2
                elements += ["H","H"]
                tmp = np.zeros([N_atoms+2,N_atoms+2])
                tmp[:N_atoms,:N_atoms] = adj_mat
                tmp[[-1,-2],count_i] = 1
                tmp[count_i,[-1,-2]] = 1
                adj_mat = tmp
                N_atoms += 2

            # Protocols for 3 missing hydrogens
            if N_expected - N_current == 3:
                if N_expected == 3:
                    rot_vec = np.array([0.0,1.0,0.0])
                    new_1 = i + np.array([H_length,0.0,0.0])
                    new_2 = axis_rot(new_1,rot_vec,i,120.0)
                    new_3 = axis_rot(new_1,rot_vec,i,240.0)
                if N_expected == 4:
                    rot_vec = normalize(cross( geo[old_inds[0]] - i, np.array([np.random.random(),np.random.random(),np.random.random()]) ))
                    new_1 = H_length*normalize(axis_rot(geo[old_inds[0]],rot_vec,i,109.5)-i) + i
                    new_2 = axis_rot(new_1,normalize(i-geo[old_inds[0]]),i,120.0)
                    new_3 = axis_rot(new_2,normalize(i-geo[old_inds[0]]),i,120.0)

                # Update geometry, adj_mat, elements, and atomtypes with three new atoms
                geo = np.vstack([geo,new_1])
                geo = np.vstack([geo,new_2])
                geo = np.vstack([geo,new_3])
                atomtypes += ["[1[{}]]".format(atomtypes[count_i].split(']')[0].split('[')[1])]*3
                elements += ["H","H","H"]
                tmp = np.zeros([N_atoms+3,N_atoms+3])
                tmp[:N_atoms,:N_atoms] = adj_mat
                tmp[[-1,-2,-3],count_i] = 1
                tmp[count_i,[-1,-2,-3]] = 1
                adj_mat = tmp
                N_atoms += 3

            # Protocols for 4 missing hydrogens
            if N_expected - N_current == 4:
                if N_expected == 4:
                    new_1 = i + np.array([H_length,0.0,0.0])
                    rot_vec = normalize(cross( new_1 - i, np.array([np.random.random(),np.random.random(),np.random.random()]) ))
                    new_2 = H_length*normalize(axis_rot(new_1,rot_vec,i,109.5)-i) + i
                    new_3 = axis_rot(new_2,normalize(i-new_1),i,120.0)
                    new_4 = axis_rot(new_3,normalize(i-new_1),i,120.0)
                    
                # Update geometry, adj_mat, elements, and atomtypes with three new atoms
                geo = np.vstack([geo,new_1])
                geo = np.vstack([geo,new_2])
                geo = np.vstack([geo,new_3])
                geo = np.vstack([geo,new_4])
                atomtypes += ["[1[{}]]".format(atomtypes[count_i].split(']')[0].split('[')[1])]*4
                elements += ["H","H","H","H"]
                tmp = np.zeros([N_atoms+4,N_atoms+4])
                tmp[:N_atoms,:N_atoms] = adj_mat
                tmp[[-1,-2,-3,-4],count_i] = 1
                tmp[count_i,[-1,-2,-3,-4]] = 1
                adj_mat = tmp
                N_atoms += 4

    if retype is True:
        return geo,id_types(elements,adj_mat,2),elements,adj_mat,range(init_len,len(geo))

    else:
        return geo,atomtypes,elements,adj_mat,range(init_len,len(geo))

# Shortcut for normalizing a vector
def normalize(x):
    return x/sum(x**(2.0))**(0.5)

# Return smiles string 
def return_smi(E,G,adj_mat=None):
    if adj_mat is None:
        xyz_write("obabel_input.xyz",E,G)
        substring = "obabel -ixyz obabel_input.xyz -ocan"
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        smile  = output.split()[0]
        os.system("rm obabel_input.xyz")
    
    else:
        mol_write("obabel_input.mol",E,G,adj_mat)
        substring = "obabel -imol obabel_input.mol -ocan"
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        smile  = output.split()[0]
        os.system("rm obabel_input.mol")

    return smile

# Return inchikey string 
def return_inchikey(E,G,adj_mat=None):
    if adj_mat is None:
        xyz_write("obabel_input.xyz",E,G)
        substring = "obabel -ixyz obabel_input.xyz -oinchikey"
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        inchi  = output.split()[0]
        os.system("rm obabel_input.xyz")
    
    else:
        mol_write("obabel_input.mol",E,G,adj_mat)
        substring = "obabel -imol obabel_input.mol -oinchikey"
        output = subprocess.Popen(substring,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        inchi  = output.split()[0]
        os.system("rm obabel_input.mol")

    return inchi

# Return true if idx is a ring atom
def ring_atom(adj_mat,idx,start=None,ring_size=10,counter=0,avoid_set=None,in_ring=None):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False,[]

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    if avoid_set is None:
        avoid_set = set([])
    if in_ring is None:
        in_ring=set([idx])

    # Trick: The fact that the smallest possible ring has three nodes can be used to simplify
    #        the algorithm by including the origin in avoid_set until after the second step
    if counter >= 2 and start in avoid_set:
        avoid_set.remove(start)    
    elif counter < 2 and start not in avoid_set:
        avoid_set.add(start)

    # Update the avoid_set with the current idx value
    avoid_set.add(idx)    
    
    # Loop over connections and recursively search for idx
    status = 0
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid_set ]
    #print cons,counter,start,avoid_set
    if len(cons) == 0:
        return False,[]
    elif start in cons:
        return True,in_ring
    else:
        for i in cons:
            if ring_atom(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid_set=avoid_set,in_ring=in_ring)[0] == True:
                in_ring.add(i)
                return True,in_ring
        return False,[]

if __name__ == "__main__":
   main(sys.argv[1:])

