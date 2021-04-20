import pandas as pd

def check_ring(smiles):
    if '1' in smiles:
        return True
    else:
        return False

def check_linear(smiles):
    if '1' in smiles:
        return False
    else:
        return True

def check_nan(i):
    if i > 0: return True
    else: return False

TCIT= pd.read_csv('TCIT_result.csv')
G4  = pd.read_csv('G4_result.csv')

TCIT_cyclic= TCIT[TCIT.smiles.apply(check_ring)]
TCIT_linear= TCIT[TCIT.smiles.apply(check_linear)]
G4_cyclic  = G4[G4.smiles.apply(check_ring)]
G4_linear  = G4[G4.smiles.apply(check_linear)]

# G4 result
G4_Exp_Cv = (G4.Cv_G4-G4.Cv_Exp).dropna()
print("Number of datapoints for G4_Exp_Cv is {}, with MSE/MAE= {}/{} J/mol*K".format(len(G4_Exp_Cv),G4_Exp_Cv.mean(),G4_Exp_Cv.abs().mean() ))
G4_Exp_S0 = (G4.S0_G4-G4.S0_Exp).dropna()
print("Number of datapoints for G4_Exp_S0 is {}, with MSE/MAE= {}/{} J/mol*K".format(len(G4_Exp_S0),G4_Exp_S0.mean(),G4_Exp_S0.abs().mean() ))

# TCIT result
TCIT_Exp_Cv = (TCIT.Cv_TCIT-TCIT.Cv_Exp).dropna()
print("Number of datapoints for TCIT_Exp_Cv is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_Exp_Cv),TCIT_Exp_Cv.mean(),TCIT_Exp_Cv.abs().mean() ))
TCIT_Exp_S0 = (TCIT.S0_TCIT-TCIT.S0_Exp).dropna()
print("Number of datapoints for TCIT_Exp_S0 is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_Exp_S0),TCIT_Exp_S0.mean(),TCIT_Exp_S0.abs().mean() ))
TCIT_G4_Cv = (TCIT.Cv_TCIT-TCIT.Cv_G4).dropna()
print("Number of datapoints for TCIT_G4_Cv is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_G4_Cv),TCIT_G4_Cv.mean(),TCIT_G4_Cv.abs().mean() ))
TCIT_G4_S0 = (TCIT.S0_TCIT-TCIT.S0_G4).dropna()
print("Number of datapoints for TCIT_G4_S0 is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_G4_S0),TCIT_G4_S0.mean(),TCIT_G4_S0.abs().mean() ))
TCIT_G4_Gf = (TCIT.Gf_TCIT-TCIT.Gf_G4).dropna()
print("Number of datapoints for TCIT_G4_Gf is {}, with MSE/MAE= {}/{} kJ/mol".format(len(TCIT_G4_Gf),TCIT_G4_Gf.mean(),TCIT_G4_Gf.abs().mean() ))

# further partition
G4_Exp_Cv = (G4_cyclic.Cv_G4-G4_cyclic.Cv_Exp).dropna()
print("Number of datapoints for G4_Exp_Cv(cyclic) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(G4_Exp_Cv),G4_Exp_Cv.mean(),G4_Exp_Cv.abs().mean() ))
G4_Exp_S0 = (G4_cyclic.S0_G4-G4_cyclic.S0_Exp).dropna()
print("Number of datapoints for G4_Exp_S0(cyclic) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(G4_Exp_S0),G4_Exp_S0.mean(),G4_Exp_S0.abs().mean() ))
G4_Exp_Cv = (G4_linear.Cv_G4-G4_linear.Cv_Exp).dropna()
print("Number of datapoints for G4_Exp_Cv(linear) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(G4_Exp_Cv),G4_Exp_Cv.mean(),G4_Exp_Cv.abs().mean() ))
G4_Exp_S0 = (G4_linear.S0_G4-G4_linear.S0_Exp).dropna()
print("Number of datapoints for G4_Exp_S0(linear) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(G4_Exp_S0),G4_Exp_S0.mean(),G4_Exp_S0.abs().mean() ))

TCIT_Exp_Cv = (TCIT_cyclic.Cv_TCIT-TCIT_cyclic.Cv_Exp).dropna()
print("Number of datapoints for TCIT_Exp_Cv(cyclic) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_Exp_Cv),TCIT_Exp_Cv.mean(),TCIT_Exp_Cv.abs().mean() ))
TCIT_Exp_S0 = (TCIT_cyclic.S0_TCIT-TCIT_cyclic.S0_Exp).dropna()
print("Number of datapoints for TCIT_Exp_S0(cyclic) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_Exp_S0),TCIT_Exp_S0.mean(),TCIT_Exp_S0.abs().mean() ))
TCIT_Exp_Cv = (TCIT_linear.Cv_TCIT-TCIT_linear.Cv_Exp).dropna()
print("Number of datapoints for TCIT_Exp_Cv(linear) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_Exp_Cv),TCIT_Exp_Cv.mean(),TCIT_Exp_Cv.abs().mean() ))
TCIT_Exp_S0 = (TCIT_linear.S0_TCIT-TCIT_linear.S0_Exp).dropna()
print("Number of datapoints for TCIT_Exp_S0(linear) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_Exp_S0),TCIT_Exp_S0.mean(),TCIT_Exp_S0.abs().mean() ))

TCIT_G4_Cv = (TCIT_cyclic.Cv_TCIT-TCIT_cyclic.Cv_G4).dropna()
print("Number of datapoints for TCIT_G4_Cv(cyclic) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_G4_Cv),TCIT_G4_Cv.mean(),TCIT_G4_Cv.abs().mean() ))
TCIT_G4_S0 = (TCIT_cyclic.S0_TCIT-TCIT_cyclic.S0_G4).dropna()
print("Number of datapoints for TCIT_G4_S0(cyclic) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_G4_S0),TCIT_G4_S0.mean(),TCIT_G4_S0.abs().mean() ))
TCIT_G4_Cv = (TCIT_linear.Cv_TCIT-TCIT_linear.Cv_G4).dropna()
print("Number of datapoints for TCIT_G4_Cv(linear) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_G4_Cv),TCIT_G4_Cv.mean(),TCIT_G4_Cv.abs().mean() ))
TCIT_G4_S0 = (TCIT_linear.S0_TCIT-TCIT_linear.S0_G4).dropna()
print("Number of datapoints for TCIT_G4_S0(linear) is {}, with MSE/MAE= {}/{} J/mol*K".format(len(TCIT_G4_S0),TCIT_G4_S0.mean(),TCIT_G4_S0.abs().mean() ))

#### generate txt files #####
TCIT_Exp_Cv_index=(TCIT.Cv_TCIT-TCIT.Cv_Exp).dropna().index
diff=[]
with open("data/TCIT_Exp_Cv.txt",'w') as f:
    f.write("{:<60s} {:<10s} {:<10s} {:<10s}\n".format("Smiles","TCIT_Cv","Exp_Cv","Dev"))
    for i in TCIT_Exp_Cv_index:
        item = TCIT.iloc[i]
        diff += [item.Cv_TCIT-item.Cv_Exp]
        f.write("{:<60s} {:<10.2f} {:<10.2f} {:<10.2f}\n".format(item.smiles,item.Cv_TCIT,item.Cv_Exp,item.Cv_TCIT-item.Cv_Exp))

TCIT_Exp_S0_index=(TCIT.S0_TCIT-TCIT.S0_Exp).dropna().index
diff=[]
with open("data/TCIT_Exp_S0.txt",'w') as f:
    f.write("{:<60s} {:<10s} {:<10s} {:<10s}\n".format("Smiles","TCIT_S0","Exp_S0","Dev"))
    for i in TCIT_Exp_S0_index:
        item = TCIT.iloc[i]
        diff += [item.S0_TCIT-item.S0_Exp]
        f.write("{:<60s} {:<10.2f} {:<10.2f} {:<10.2f}\n".format(item.smiles,item.S0_TCIT,item.S0_Exp,item.S0_TCIT-item.S0_Exp))

TCIT_G4_Cv_index=(TCIT.Cv_TCIT-TCIT.Cv_G4).dropna().index
diff=[]
with open("data/TCIT_G4_Cv.txt",'w') as f:
    f.write("{:<60s} {:<10s} {:<10s} {:<10s}\n".format("Smiles","TCIT_Cv","G4_Cv","Dev"))
    for i in TCIT_G4_Cv_index:
        item = TCIT.iloc[i]
        diff += [item.Cv_TCIT-item.Cv_G4]
        f.write("{:<60s} {:<10.2f} {:<10.2f} {:<10.2f}\n".format(item.smiles,item.Cv_TCIT,item.Cv_G4,item.Cv_TCIT-item.Cv_G4))

TCIT_G4_S0_index=(TCIT.S0_TCIT-TCIT.S0_G4).dropna().index
diff=[]
with open("data/TCIT_G4_S0.txt",'w') as f:
    f.write("{:<60s} {:<10s} {:<10s} {:<10s}\n".format("Smiles","TCIT_S0","G4_S0","Dev"))
    for i in TCIT_G4_S0_index:
        item = TCIT.iloc[i]
        diff += [item.S0_TCIT-item.S0_G4]
        f.write("{:<60s} {:<10.2f} {:<10.2f} {:<10.2f}\n".format(item.smiles,item.S0_TCIT,item.S0_G4,item.S0_TCIT-item.S0_G4))

TCIT_G4_Gf_index=(TCIT.Gf_TCIT-TCIT.Gf_G4).dropna().index
diff=[]
with open("data/TCIT_G4_Gf.txt",'w') as f:
    f.write("{:<60s} {:<10s} {:<10s} {:<10s}\n".format("Smiles","TCIT_Gf","G4_Gf","Dev"))
    for i in TCIT_G4_Gf_index:
        item = TCIT.iloc[i]
        diff += [item.Gf_TCIT-item.Gf_G4]
        f.write("{:<60s} {:<10.2f} {:<10.2f} {:<10.2f}\n".format(item.smiles,item.Gf_TCIT,item.Gf_G4,item.Gf_TCIT-item.Gf_G4))

G4_Exp_Cv_index=(G4.Cv_G4-G4.Cv_Exp).dropna().index
diff=[]
with open("data/G4_Exp_Cv.txt",'w') as f:
    f.write("{:<60s} {:<10s} {:<10s} {:<10s}\n".format("Smiles","G4_Cv","Exp_Cv","Dev"))
    for i in G4_Exp_Cv_index:
        item = G4.iloc[i]
        diff += [item.Cv_G4-item.Cv_Exp]
        f.write("{:<60s} {:<10.2f} {:<10.2f} {:<10.2f}\n".format(item.smiles,item.Cv_G4,item.Cv_Exp,item.Cv_G4-item.Cv_Exp))

G4_Exp_S0_index=(G4.S0_G4-G4.S0_Exp).dropna().index
diff=[]
with open("data/G4_Exp_S0.txt",'w') as f:
    f.write("{:<60s} {:<10s} {:<10s} {:<10s}\n".format("Smiles","G4_S0","Exp_S0","Dev"))
    for i in G4_Exp_S0_index:
        item = G4.iloc[i]
        diff += [item.S0_G4-item.S0_Exp]
        f.write("{:<60s} {:<10.2f} {:<10.2f} {:<10.2f}\n".format(item.smiles,item.S0_G4,item.S0_Exp,item.S0_G4-item.S0_Exp))
