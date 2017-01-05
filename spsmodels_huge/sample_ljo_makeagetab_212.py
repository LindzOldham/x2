import numpy as np, pylab as pl, pyfits as py, cPickle
from linslens.ClipResult import clipburnin
from astLib import astCalc

names = py.open('/data/ljo31/Lens/LensParams/Phot_1src_huge_new.fits')[1].data['name']
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshiftsUpdated_1.00_source.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated_1.00_lens.npy')[()]

srcages, lensages = [],[]
dsrcages, dlensages = [],[]
masses = np.zeros((6,len(names)))
ii = 0
for name in names: 
    lo,med,hi=np.load('/data/ljo31b/EELs/inference/new/huge/212_params_'+name+'.npy').T
    z_lens = lz[name][0]
    z_src = sz[name][0]
    lensage = med[-3]+astCalc.tl(z_lens)
    srcage = med[3]+astCalc.tl(z_src)
    lolens = med[-3] - lo[-3]
    hilens = hi[-3]-med[-3]
    losrc = med[3]-lo[3]
    hisrc = hi[3]-med[3]
    print name, lensage, srcage
    srcages.append(srcage)
    lensages.append(lensage)
    dsrcages.append(np.mean((losrc,hisrc)))
    dlensages.append(np.mean((lolens,hilens)))
    

np.save('/data/ljo31b/EELs/inference/new/huge/212_ages',[lensages,srcages,dlensages,dsrcages])
