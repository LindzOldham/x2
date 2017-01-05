import numpy as np, pylab as pl

magnifications = np.load('/data/ljo31/Lens/LensParams/magnifications_212_huge.npy')[()]
names = magnifications.keys()
names.sort()

oldmasses = np.load('/data/ljo31b/EELs/inference/new/huge/masses_212.npy')

newmasses = oldmasses*0.
old = oldmasses[3]
dold = np.mean((oldmasses[4],oldmasses[5]),0)

n=0
for name in names:
    newMs = np.load('/data/ljo31b/EELs/inference/new/huge/NEWFORFP_212_masses_'+name+'.npy')
    new,dnew = newMs[3],np.mean((newMs[5],newMs[4]),0)
    print name, '%.2f'%old[n], '$\pm$', '%.2f'%dold[n], '&', '%.2f'%new, '$\pm$', '%.2f'%dnew, r'\\'
    newmasses[:,n] = newMs
    n+=1

np.save('/data/ljo31b/EELs/inference/new/huge/NEWFORFPNEW_masses_212.npy',newmasses)
