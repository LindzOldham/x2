import numpy as np, pylab as pl, pyfits as py, cPickle
from esi_indous_vdfit import run

names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','jul2016_J1605','J1619','jul2016_J2228']

sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]


for name in names:
    run(lz[name],sz[name],fit=False,read=True,File = filename,typ=typ,wid=wid)
    pl.show()
