import numpy as np, pylab as pl

magnifications = np.load('/data/ljo31/Lens/LensParams/magnifications_212_huge.npy')[()]
names = magnifications.keys()
names.sort()


newmasses = np.zeros((3,13))

n=0
for name in names:
    result = np.load('/data/ljo31b/EELs/inference/new/huge/result_212_CHECK_SALPETER_'+name)
    lp,trace,dic,_ = result
    m = dic['massL']
    mass = np.median(m)
    lo,hi = mass-np.percentile(m,16), np.percentile(m,84)-mass
    print name, mass, lo, hi
    newmasses[:,n] = [mass,lo,hi]
    n+=1

np.save('/data/ljo31b/EELs/inference/new/huge/salpeter_masses_212.npy',newmasses)
