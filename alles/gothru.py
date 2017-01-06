import numpy as np, pylab as pl, pyfits as py, cPickle
from linslens import EELsImages as E

for name in ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1606','J1619','J2228']:
    print name
    try:
        result = np.load('/data/ljo31/Lens/LensModels/'+name+'_211')
    except:
        continue
    lp,trace,dic,_ = result
    print 'source 1 re', '%.2f'%np.median(dic['Source 1 re'][:,0].ravel())
    print 'galaxy 1 re','%.2f'%np.median(dic['Galaxy 1 re'][:,0].ravel())
    if 'Galaxy 2 re' in dic.keys():
        print 'galaxy 2 re','%.2f'%np.median(dic['Galaxy 2 re'][:,0].ravel())
    
