from keras.layers import Dense, Lambda,Input, concatenate,add
from keras import optimizers
from keras import regularizers
import keras.models as models
from keras import backend as K
from keras.layers.normalization import BatchNormalization

degrees = range(1,5) # This refers to the number of bonds a given atom may have. Lower bound is always 1, upper bound specified by the dataset. 4 is probably fine for most standard datasets

def neural_conv_layer(inputs, atom_features_of_previous_layer, num_atom_features,
                      conv_width, fp_length, L2_reg, num_bond_features,
                      batch_normalization=False, layer_index = 0,input_index=0):
#    atom_features_of_previous_layer # either (variable_a, num_input_atom_features) [first layer] or (variable_a, conv_width)
    

    activations_by_degree = []


    for degree in degrees:
        
        atom_features_of_previous_layer_x_degree = K.sum(K.gather(atom_features_of_previous_layer, indices=inputs['atom_neighbors_indices_{}_degree_{}'.format(input_index,degree)]), 1) #Collects the atom features of degree x from the previous layer and collapses them  along the columns, gives a vectore of length = fp_length

        merged_atom_bond_features = concatenate([atom_features_of_previous_layer_x_degree, inputs['bond_features_{}_degree_{}'.format(input_index,degree)]], axis=1)

        merged_atom_bond_features_shape = (None,num_atom_features+num_bond_features)

        activations = Dense(conv_width, activation = 'relu', bias = False, name = 'activations_{}_{}_degree_{}'.format(input_index,layer_index,degree))(merged_atom_bond_features)

        activations_by_degree.append(activations)
            
    #Skip-connection to output/final fingerprint

    output_to_fingerprint_tmp = Dense(fp_length,activation='softmax',name='fingerprint_{}_skip_connection_{}'.format(input_index,layer_index))(atom_features_of_previous_layer) # shape = (variable, fp_length)

    def my_dot(y):
        return K.dot(inputs['atom_batch_matching_{}_degree_{}'.format(input_index,degree)],y)
    output_to_fingerprint = Lambda(my_dot)(output_to_fingerprint_tmp)

#    output_to_fingerprint = Lambda(lambda x: K.dot(inputs['atom_batch_matching_{}_degree_{}'.format(input_index,degree)],x))(output_to_fingerprint_tmp) #shape = (batch_size,fp_length)

    # connect to next layer

    this_activation_tmp = Dense(conv_width,activation='relu',name = 'layer_{}_{}_activations'.format(input_index,layer_index))(atom_features_of_previous_layer) #shape = (variable_a,conv_width)

    merged_neighbor_activations = concatenate(activations_by_degree,axis=0) #shape = (variable, conv_width)
            

    def my_add(y):
        return merged_neighbor_activations+y
    new_atom_features = Lambda(my_add)(this_activation_tmp)

#    new_atom_features = Lambda(lambda x: merged_neighbor_activations + x)(this_activation_tmp)

    if batch_normalization:
        new_atom_features = BatchNormalization()(new_atom_features)

        

    return new_atom_features,output_to_fingerprint

def build_fp_model(fp_length = 50, fp_depth=3,conv_width=30,predictor_MLP_layers = [200,200,200],
                   L2_reg=4e-4,num_input_atom_features=62,
                   num_bond_features=6,batch_normalization=False,lr=1e-4,input_size=1):

    """
    fp_length   # Usually neural fps need far fewer dimensions than morgan.
    fp_depth     # The depth of the network equals the fingerprint radius.
    conv_width   # Only the neural fps need this parameter.
    h1_size     # Size of hidden layer of network on top of fps.
    
    """

    ###
    #Neural fingerprint block
    ###
    
    inputs = {}
    
    fingerprints = []

    for i in range(input_size):
        inputs['input_atom_features_{}'.format(i)]=Input(name ='input_atom_features_{}'.format(i),shape=(num_input_atom_features,))
        for degree in degrees:
            inputs['bond_features_{}_degree_{}'.format(i,degree)] = Input(name='bond_features_{}_degree_{}'.format(i,degree),shape=(num_bond_features,))
            inputs['atom_neighbors_indices_{}_degree_{}'.format(i,degree)] = Input(name='atom_neighbors_indices_{}_degree_{}'.format(i,degree),shape=(degree,),dtype='int32')

            inputs['atom_batch_matching_{}_degree_{}'.format(i,degree)]=Input(name='atom_batch_matching_matrix_{}_degree_{}'.format(i,degree),shape=(None,)) #shape = (batch_size,variable)
#            print('Degree: {}'.format(degree))
        atom_features = inputs['input_atom_features_{}'.format(i)]
        all_outputs_to_fingerprint = []
        num_atom_features = num_input_atom_features

        for j in range(fp_depth):
            atom_features,output_to_fingerprint = neural_conv_layer(inputs,atom_features_of_previous_layer=atom_features,num_atom_features=num_atom_features,conv_width=conv_width,fp_length=fp_length,L2_reg=L2_reg,num_bond_features=num_bond_features,batch_normalization=batch_normalization,layer_index=j,input_index=i)
            num_atom_features=conv_width

            all_outputs_to_fingerprint.append(output_to_fingerprint)

        neural_fingerprint = add(all_outputs_to_fingerprint) if len(all_outputs_to_fingerprint)>1 else all_outputs_to_fingerprint
    
        fingerprints.append(neural_fingerprint)
    

    prediction_mlp_layer = add(fingerprints) if len(fingerprints)>1 else neural_fingerprint

    for layer,units in enumerate(predictor_MLP_layers):
        prediction_mlp_layer = Dense(units,activation='relu',W_regularizer=regularizers.l2(L2_reg),name='MLP_hidden_'+str(layer))(prediction_mlp_layer)

    output = Dense(1,activation='linear',name='prediction')(prediction_mlp_layer)

    model = models.Model(input = list(inputs.values()),output = [output])
    model.compile(optimizer= optimizers.RMSprop(learning_rate=lr), loss = {'prediction':'mse'})
    return model


def build_fp_model_concat(fp_length = 50, fp_depth=3,conv_width=30,predictor_MLP_layers = [200,200,200],
                   L2_reg=4e-4,num_input_atom_features=62,
                   num_bond_features=6,batch_normalization=False,input_size=1):

    """
    fp_length   # Usually neural fps need far fewer dimensions than morgan.
    fp_depth     # The depth of the network equals the fingerprint radius.
    conv_width   # Only the neural fps need this parameter.
    h1_size     # Size of hidden layer of network on top of fps.
    
    """

    ###
    #Neural fingerprint block
    ###
    
    inputs = {}
    
    fingerprints = []

    for i in range(input_size):
        inputs['input_atom_features_{}'.format(i)]=Input(name ='input_atom_features_{}'.format(i),shape=(num_input_atom_features,))
        for degree in degrees:
            inputs['bond_features_{}_degree_{}'.format(i,degree)] = Input(name='bond_features_{}_degree_{}'.format(i,degree),shape=(num_bond_features,))
            inputs['atom_neighbors_indices_{}_degree_{}'.format(i,degree)] = Input(name='atom_neighbors_indices_{}_degree_{}'.format(i,degree),shape=(degree,),dtype='int32')

            inputs['atom_batch_matching_{}_degree_{}'.format(i,degree)]=Input(name='atom_batch_matching_matrix_{}_degree_{}'.format(i,degree),shape=(None,)) #shape = (batch_size,variable)
#            print('Degree: {}'.format(degree))
        atom_features = inputs['input_atom_features_{}'.format(i)]
        all_outputs_to_fingerprint = []
        num_atom_features = num_input_atom_features

        for j in range(fp_depth):
            atom_features,output_to_fingerprint = neural_conv_layer(inputs,atom_features_of_previous_layer=atom_features,num_atom_features=num_atom_features,conv_width=conv_width,fp_length=fp_length,L2_reg=L2_reg,num_bond_features=num_bond_features,batch_normalization=batch_normalization,layer_index=j,input_index=i)
            num_atom_features=conv_width

            all_outputs_to_fingerprint.append(output_to_fingerprint)

        neural_fingerprint = add(all_outputs_to_fingerprint) if len(all_outputs_to_fingerprint)>1 else all_outputs_to_fingerprint
    
        fingerprints.append(neural_fingerprint)
    

    prediction_mlp_layer = concatenate(fingerprints,axis=1) if len(fingerprints)>1 else neural_fingerprint

    for layer,units in enumerate(predictor_MLP_layers):
        prediction_mlp_layer = Dense(units,activation='relu',W_regularizer=regularizers.l2(L2_reg),name='MLP_hidden_'+str(layer))(prediction_mlp_layer)

    output = Dense(1,activation='linear',name='prediction')(prediction_mlp_layer)

    model = models.Model(input = list(inputs.values()),output = [output])
    model.compile(optimizer= optimizers.Adam(learning_rate=1e-4), loss = {'prediction':'mse'})
    return model
