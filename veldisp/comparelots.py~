import numpy as np, pylab as pl, pyfits as py
import sys
import glob


def plot(name):
    infile = '/data/ljo31b/EELs/esi/kinematics/inference/wide/'+name
    result = np.load(infile)

    lp,trace,dic,_=result
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/'+name)
    lp2,trace2,dic2,_=result


    for key in 'lens dispersion','source dispersion':
        pl.figure()
        pl.title(key)
        pl.hist(dic[key][300:].ravel(),30,alpha=0.5,histtype='stepfilled',normed=True)
        pl.hist(dic2[key][200:].ravel(),30,alpha=0.5,histtype='stepfilled',normed=True)
    pl.title(name)
    pl.show()


names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1248','J1323','J1347','J1446','J1605','J1606','J1619','J2228']

for name in names:
    plot(name)

