import numpy,pylab,pyfits
import vdfit as vd
import sys
from spectra import resolution,spectools as st

tmps = vd.INDOTEMPS
tmps = vd.INDO_AF
out = {}

img = '/data/ljo31b/EELs/esi/kinematics/apertures/final/J0837_ap_1.00_spec_lens.fits'
var = '/data/ljo31b/EELs/esi/kinematics/apertures/final/J0837_ap_1.00_var_lens.fits'
wave = '/data/ljo31b/EELs/esi/kinematics/apertures/final/J0837_ap_1.00_wl_lens.fits'
img = pyfits.open(img)[0].data
var = pyfits.open(var)[0].data
wave = pyfits.open(wave)[0].data

print numpy.isnan(img).sum(),numpy.isnan(var).sum()


#st.make_spec(img,var,wave,'J0837.fits')
img = 'J0837.fits'


res = resolution.get_resolution(img)
print 'res',res
spec = pyfits.open(img)[1].data#**0.5
wave = st.wavelength(img,1)
#pylab.plot(wave,spec)
#pylab.show()

res = 21.
redshift = 0.42485
vdisp = 200.

l1 = 4055.
l2 = 4900.

l1 = 3850.
l2 = 4450.
#l2 = 6000.
#l2 = 4700.
tres = 1.2*299792/(0.5*(l1+l2))
tres /= 2.355

rmask = [[4500,4600.]]
omask = [[6855.,6925.]]
a = vd.vd.pipeline(img,tmps,redshift,res,tres,[[l1,l2]],sigma=vdisp,nfit=5,niter=1000,smax=601.,rmask=rmask,omask=omask)
print a['sigma'],a['vel'],a['errors'],redshift
redshift -= a['vel']/299792.
a = vd.vd.pipeline(img,tmps,redshift,res,tres,[[l1,l2]],sigma=vdisp,nfit=3,niter=5000,smax=351.,rmask=rmask,omask=omask)
print a['sigma'],a['vel'],a['errors'],redshift
redshift -= a['vel']/299792.
a = vd.vd.pipeline(img,tmps,redshift,res,tres,[[l1,l2]],sigma=vdisp,nfit=5,niter=5000,smax=351.,rmask=rmask,omask=omask)
print a['sigma'],a['vel'],a['errors'],redshift
#a = vd.vd.pipeline(img,tmps,redshift,res,tres,[[l1,l2]],sigma=vdisp,nfit=4,niter=5000,smax=251.,rmask=rmask,omask=omask)
print a['sigma'],a['vel'],a['errors'],redshift

vd.plot(a)
pl
