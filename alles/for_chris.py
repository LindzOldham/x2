import pyfits as py
import numpy as np

vkidata = np.load('/data/ljo31/Lens/LensParams/VIK_phot_212_dict_huge_new.npy')[()] # make this
lzs = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]
keys = lzs.keys()
keys.sort()

cat = []

for name in keys:
    if name == 'J1248':
        continue
    print name
    v_src,i_src,dv_src,di_src,vi_src,dvi_src, v_lens,i_lens,dv_lens,di_lens,vi_lens,dvi_lens, k_src, dk_src, k_lens, dk_lens = vkidata[name]
    print lzs[name][0], k_lens
    cat.append([name,[lzs[name][0],i_lens,di_lens]])

cat = dict(cat)
np.save('/data/ljo31/Lens/LensParams/z_I_dI_lenses',cat)

for name in cat.keys():
    z,I,dI = cat[name]

