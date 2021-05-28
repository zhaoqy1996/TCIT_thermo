## TCIT-Hf

**TCIT**, the short of Taffi component increment theory, is a powerful tool to predict thermochemistry properties, like enthalpy of formation.

This script implemented TCIT which performs on a given folder of target compounds based on a fixed TCIT CAV database distributed with the paper "[A Self-Consistent Component Increment Theory for Predicting Enthalpy of Formation](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00092)" by Zhao and Savoie. Further ring correction is added distributed with the paper "[Transferable Ring Corrections for Predicting Enthalpy of Formation of Cyclic Compounds](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00367)" by Zhao, Iovanac and Savoie. The heat of vaporization wil take NIST value is available, otherwis it will be computed by SIMPOL1 model ("[Vapor pressure prediction – simple group contribution method](https://doi.org/10.5194/acp-8-2773-2008)"). Similarly, the heat of sublimation comes from "[Simple yet accurate prediction method for sublimation enthalpies of organic contaminants using their molecular structure](https://doi.org/10.1016/j.tca.2012.05.008)" 

The script operates on either a folder of xyz files or a list of smiles strings, prints out the Taffi components and corresponding CAVs that are used for each prediction, and returns the 0K and 298K enthalpy of formation as well as enthalpy of formation of liquid phase and solid phase. 

## Software requirement
1. openbabel 2.4.1 or higher
2. anaconda

## Set up an environment if needed
* conda create -n TCIT -c conda-forge python=3.7 rdkit
* source activate TCIT
* pip install alfabet
* pip install mordred

## Usage
If your input type a xyz file:

1. Put xyz files of the compounds with research interest in one folder (default: input_xyz)
2. Type "python TCIT.py -h" for help if you want specify the database files and run this program.
3. By default, run "python TCIT" and the program will take all xyz files in "input_xyz" and return a prediction result in result.log and result_output.txt.

If your input type is smiles string:

1. Make a list of smiles string (default: input_list/test_inp.txt)
2. Type "python TCIT.py -h" for help if you want specify the database files and run this program.
3. By default, run "python TCIT -t smiles" and the program will take all smiles string in input_list/test_inp.txt and return a prediction result in result.log and result_output.txt.

If you want to get liquid/solid phase predictions as well, add --liquid/--solid in the end. 

e.g. 'python TCIT.py -t smiles -i input_list/test_inp.txt --liquid --solid'

## Notes
1. Make sure the bin folder of openbabel is added in the environment setting, or 'alias obabel=' to that bin folder. Check by running 'obabel -H'.
2. Currently TCIT just works for Linux and MacOS.
