import pyfits as py, numpy as np, pylab as pl

K = py.open('/data/ljo31/Lens/J1144/J1144_Kp_narrow.fits')[0].data.copy()[538:895,324:751]
header = py.open('/data/ljo31/Lens/J1144/J1144_Kp_narrow.fits')[0].header.copy()
py.writeto('/data/ljo31/Lens/J1144/J1144_Kp_narrow_cutout.fits',K,header)

