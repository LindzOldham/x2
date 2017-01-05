import numpy as np, pylab as pl, pyfits as py
import cPickle

phot = np.load('/data/ljo31/Lens/LensParams/Structure_2src.npy')[0]

for name in phot.keys():
    if 'Source 2 n' in phot[name].keys():
        print name, '&','%.2f'%phot[name]['Source 1 q'], '&','%.2f'%phot[name]['Source 2 q'], '&','%.2f'%phot[name]['Source 1 pa'], '&','%.2f'%phot[name]['Source 2 pa'], '&','%.2f'%phot[name]['Source 1 re'],'&', '%.2f'%phot[name]['Source 2 re'],r'\\'
