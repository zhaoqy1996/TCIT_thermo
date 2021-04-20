import numpy as np

Exp = []
TCIT_Exp = []
TCIT_Exp_dev = []

with open('compare.txt','r') as f:
    
    for lc,lines in enumerate(f):
        fields = lines.split()

        if len(fields) < 3: continue
        Exp += [float(fields[2])]
        TCIT_Exp += [float(fields[1])-float(fields[2])]
        TCIT_Exp_dev += [abs(float(fields[1])-float(fields[2])) / float(fields[2])]

TCIT_Exp = np.array(TCIT_Exp)
TCIT_Exp_dev = np.array(TCIT_Exp_dev)

print(min(Exp),max(Exp))
print(np.mean(TCIT_Exp ))
print(np.mean(abs(TCIT_Exp)))
print(np.mean(TCIT_Exp_dev))
exit()
Exp_Cv = []
TCIT_Exp_Cv = []

with open('cyclic_TCIT_Exp_Cv.txt','r') as f:
    
    for lc,lines in enumerate(f):
        fields = lines.split()

        if len(fields) < 3: continue
        Exp_Cv += [float(fields[2])]
        TCIT_Exp_Cv += [float(fields[1])-float(fields[2])]

TCIT_Exp_Cv = np.array(TCIT_Exp_Cv)
print(min(Exp_Cv),max(Exp_Cv))
print(np.mean(TCIT_Exp_Cv ))
print(np.mean(abs(TCIT_Exp_Cv)))


G4_S0 = []
TCIT_G4_S0 = []

with open('cyclic_TCIT_G4_S0.txt','r') as f:
    
    for lc,lines in enumerate(f):
        fields = lines.split()

        if len(fields) < 3: continue
        G4_S0 += [float(fields[2])]
        TCIT_G4_S0 += [float(fields[1])-float(fields[2])]

TCIT_G4_S0 = np.array(TCIT_G4_S0)
print(min(G4_S0),max(G4_S0))
print(np.mean(TCIT_G4_S0 ))
print(np.mean(abs(TCIT_G4_S0)))

G4_Cv = []
TCIT_G4_Cv = []

with open('cyclic_TCIT_G4_Cv.txt','r') as f:
    
    for lc,lines in enumerate(f):
        fields = lines.split()

        if len(fields) < 3: continue
        G4_Cv += [float(fields[2])]
        TCIT_G4_Cv += [float(fields[1])-float(fields[2])]

TCIT_G4_Cv = np.array(TCIT_G4_Cv)
print(min(G4_Cv),max(G4_Cv))
print(np.mean(TCIT_G4_Cv ))
print(np.mean(abs(TCIT_G4_Cv)))


exit()
# initialize dictionary
G4_TCIT_Cv_diff = []
G4_TCIT_S0_diff = []

#with open()

Exp_TCIT_Cv_corr_diff = []
Exp_TCIT_Cv_corr_dev  = []
Exp_TCIT_Cv_diff = []
Exp_TCIT_Cv_dev  = []

with open('TCIT_corr_Exp_Cv.txt','r') as f:
    for lc,lines in enumerate(f):
        if lc == 0: continue

        fields = lines.split()
        Exp_TCIT_Cv_corr_diff += [float(fields[2]) - float(fields[3])]
        Exp_TCIT_Cv_corr_dev  += [abs(float(fields[2])- float(fields[3]))/float(fields[3])]
        Exp_TCIT_Cv_diff += [float(fields[1]) - float(fields[3])]
        Exp_TCIT_Cv_dev  += [abs(float(fields[1])- float(fields[3]))/float(fields[3])]

Exp_TCIT_Cv_corr_diff = np.array(Exp_TCIT_Cv_corr_diff)
Exp_TCIT_Cv_corr_dev  = np.array(Exp_TCIT_Cv_corr_dev)
Exp_TCIT_Cv_diff = np.array(Exp_TCIT_Cv_diff)
Exp_TCIT_Cv_dev  = np.array(Exp_TCIT_Cv_dev)
#print(max(Exp_TCIT_Cv_corr_diff))
print(np.mean(Exp_TCIT_Cv_corr_diff ))
print(np.mean(abs(Exp_TCIT_Cv_corr_diff )))
print(np.mean(Exp_TCIT_Cv_corr_dev))
print(np.mean(Exp_TCIT_Cv_diff ))
print(np.mean(abs(Exp_TCIT_Cv_diff )))
print(np.mean(Exp_TCIT_Cv_dev))

