import utilities
import numpy as np
degrees = range(1,5)
def neuralProcess(smiles):
    """
    Its fairly tough to get the data into a format that
    Keras can split into batches on its own with the .fit
    comman. Instead, for this particular model its a lot easier
    to split the data up ourselves and train on batch. That also
    means that you'll need to evaluate on batch as well
    """

    num_bond_features = 6 #single, double, triple, aromatic, conjugation, cyclic

    input_dict = {}
    for s,smile_list in enumerate(smiles):

        array_rep = utilities.array_rep_from_smiles(smile_list)
        atom_features = array_rep['atom_features'] 
        summed_bond_features_by_degree = bond_features_by_degree(array_rep)
    
        input_dict['input_atom_features_{}'.format(s)]=atom_features #Maybe I can add the support for multiple SMILES here
        missing_degrees = []

        for degree in degrees:
            atom_neighbors_list = array_rep[('atom_neighbors',degree)]
        
            if len(atom_neighbors_list)==0:
                missing_degrees.append(degree)
                continue
        
            atom_adjacency_matrix = build_adjacency_matrix(atom_neighbors_list,atom_features.shape[0])
            atom_batch_matrix = build_adjacency_matrix(array_rep['atom_list'],atom_features.shape[0]).T
            
            assert np.all(atom_batch_matrix.sum(1).mean()==1)
            assert np.all(atom_batch_matrix.sum(0).mean()>1),'Error: looks like a single-atom molecule?'
            

            input_dict['bond_features_{}_degree_{}'.format(s,degree)] = summed_bond_features_by_degree[degree]

            input_dict['atom_neighbors_indices_{}_degree_{}'.format(s,degree)] = atom_neighbors_list
            input_dict['atom_features_selector_matrix_{}_degree_{}'.format(s,degree)] = atom_adjacency_matrix
            input_dict['atom_batch_matching_matrix_{}_degree_{}'.format(s,degree)] = atom_batch_matrix.T # (batchsize, num_atoms)
            
            if degree==0:
                print('degree 0 bond?')
                return

            num_bond_features = input_dict['bond_features_{}_degree_{}'.format(s,degree)].shape[1]
            num_atoms = atom_adjacency_matrix.shape[1]

        for missing_degree in missing_degrees:
            input_dict['atom_neighbors_indices_{}_degree_{}'.format(s,missing_degree)] = np.zeros((0, missing_degree),'int32')
            input_dict['bond_features_{}_degree_{}'.format(s,missing_degree)] = np.zeros((0, num_bond_features),'float32')
            input_dict['atom_features_selector_matrix_{}_degree_{}'.format(s,missing_degree)] = np.zeros((0, num_atoms),'float32') 
            input_dict['atom_batch_matching_matrix_{}_degree_{}'.format(s,missing_degree)] = atom_batch_matrix.T
    return input_dict

def bond_features_by_degree(array_rep):

    # Returns a list with size equal to the number of degrees, inclusive. Elements will be of shape (number of atoms of degree x, number of bond features)
    # Index 1 corresponds to the atom, index 2 to the number of bond features
    # 
    bond_features_atom_degree = []
    bond_features = array_rep['bond_features']
    for degree in range(max(degrees)+1):
        bond_neighbors_list=array_rep[('bond_neighbors',degree)]
        summed_bond_neighbors = bond_features[bond_neighbors_list].sum(axis=1) #Sums up the bond features for a given atom, across all of the atoms
        bond_features_atom_degree.append(summed_bond_neighbors)
    return bond_features_atom_degree

def build_adjacency_matrix(neighbors_lists,total_num_features):
    N = len(neighbors_lists)
    mat = np.zeros((N,total_num_features),'float32')
    for i,e in enumerate(neighbors_lists):
        mat[i,e]=1
    return mat
