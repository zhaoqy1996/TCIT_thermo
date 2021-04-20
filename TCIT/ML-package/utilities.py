import json
from keras.utils import to_categorical
from sklearn.model_selection import KFold
import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

np.random.seed(0)
degrees = range(5)

def writeCharacterSet(smiles):
    chars = set()
    for s in smiles:
        for c in s:
            if c not in chars:
                chars.add(c)
    chars.add('') #Padding Character

    c_dict = {c:i for i,c in enumerate(chars)}
    with open('characters.json','w', encoding='utf-8') as f:
        json.dump(c_dict, f, ensure_ascii=False, indent=4)

def getVector(smile,max_len):
    c_dict = readCharacterSet()
    s_vector = []
    token = tokenize(smile)

    smile = pad(token,max_len)
    for i in smile:
        s_vector.append(c_dict[i])
    

    return s_vector
    

def getOneHot(smile,max_len):
    vector = getVector(smile,max_len)
#    vector = pad(vector,max_len)
    oh = to_categorical(vector)
#    assert oh.shape == (50, 26)
    return oh


def pad(vector,max_length):
    while len(vector)<max_length:
        vector.append('')
    return vector

def readCharacterSet():
    with open('characters.json','r') as f:
        c_dict = json.load(f)
    return c_dict


def splitData(n_splits):
    kf = KFold(n_splits=n_splits)
    return kf
    
def tokenize(smiles):
    long_tokens = [
        '@@',
        'Cl',
        'Br'
        ]
    
    replacements = [
        '!',
        '$',
        '%'
        ]

    for i,e in enumerate(long_tokens):
        smiles = smiles.replace(e,replacements[i])
        
    tokens = []

    for s in smiles: #Iterate over characters in smiles string
        try:
            i = replacements.index(s) #If character is a dummy token, get index within replacement list
            tokens.append(long_tokens[i]) #Replace dummy token with correct sequence
        except ValueError:
            tokens.append(s) #If character is not a dummy token, add to output sequence
    return tokens


def getEarlyStop():
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)
    return es

def getRateDecay():
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.9,
                                  patience = 10,
                                  min_lr = 0.000001)
    return reduce_lr
    
    

def array_rep_from_smiles(smiles):
    """Compute all of the SMILES features. This is directly from the original
    chemical-GCNN implementation"""
    import gc_utils as molgraph
    molgraph = molgraph.graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] =np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] =np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep
