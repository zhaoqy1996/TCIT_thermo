## TCIT-Hf

**TCIT**, the short of Taffi component increment theory, is a powerful tool to predict thermochemistry properties, like enthalpy of formation.

This script implemented TCIT which performs on a given folder of target compounds based on a fixed TCIT CAV database distributed with the paper "A Self-Consistent Component Increment Theory for Predicting Enthalpy of Formation" by Zhao and Savoie. (The paper can be found [here](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00092?casa_token=J-tbN5mxhiAAAAAA:KaJcTVzRs0t3M3kkwdSpvg5LQkAD6iSyzpUEjzNg_MmwqNGdmah57E_NSlwBlJ81p8ROOqibqUN8NEs5).) Further ring correction is added distributed with the paper "Ring correction XXX"' by Zhao, Iovanac and Savoie. The heat of vaporization wil take NIST value is available, otherwis it will be computed by SIMPOL1 model ("Vapor pressure prediction – simple group contribution method". The paper can be found [here](https://acp.copernicus.org/articles/8/2773/2008/))

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

1. Put xyz files of the compounds with research interest in one folder (default: inputxyz)
2. Type "python TCIT.py -h" for help if you want specify the database files and run this program.
3. By default, run "python TCIT" and the program will take all xyz files in "input_xyz" and return a prediction result in result.log

If your input type is smiles string:

1. Make a list of smiles string (default: input.txt)
2. Type "python TCIT.py -h" for help if you want specify the database files and run this program.
3. By default, run "python TCIT -t smiles" and the program will take all smiles string in input.txt and return a prediction result in result.log

If you want to get liquid/solid phase predictions as well, add --liquid/--solid in the end. 

I.e. 'python TCIT.py -t smiles -i input_list/test_inp.txt --liquid --solid'

## Notes
1. Make sure the bin folder of openbabel is added in the environment setting, or 'alias obabel=' to that bin folder 
