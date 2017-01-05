import numpy as np, pylab as pl, pyfits as py
import sys
import glob


def plot(name):
    if name == 'J1605':
        return
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/apertures/'+name+'_1.50_lens')
    lp,trace,dic,_=result
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/apertures/'+name+'_1.50_source')
    lp2,trace2,dic2,_=result
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/apertures/'+name+'_1.75')
    lp3,trace3,dic3,_=result
    result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/apertures/'+name+'_1.5')
    lp4,trace4,dic4,_=result

    key = 'lens dispersion'
    pl.figure()
    pl.title(key)
    pl.hist(dic[key][300:].ravel(),30,alpha=0.5,histtype='stepfilled',normed=True)
    pl.hist(dic3[key][300:].ravel(),30,alpha=0.5,histtype='stepfilled',normed=True)
    pl.title(name+' '+key)

    key = 'source dispersion'
    pl.figure()
    pl.title(key)
    pl.hist(dic2[key][300:].ravel(),30,alpha=0.5,histtype='stepfilled',normed=True)
    pl.hist(dic3[key][300:].ravel(),30,alpha=0.5,histtype='stepfilled',normed=True)
    pl.title(name+' '+key)
    pl.show()


names = ['J1323','J1347','J1446','J1605','J1606','J1619','J2228']

for name in names:
    plot(name)

