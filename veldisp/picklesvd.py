import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from bluefitter import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g

dir = '/data/ljo31b/EELs/esi/PICKLES/'
templates = ['K3III.dat','K2III.dat','G5III.dat','K1III.dat','K0III.dat','G0III.dat','G8III.dat','A0III.dat','F2III.dat']

for i in range(len(templates)):
    templates[i] = dir+templates[i]

VGRID = 1.
light = 299792.458
ln10 = np.log(10.)



def getmodel(twave,tspec,tscale,smin=5.,smax=501):
    match = tspec.copy()
    # spectra are at higher resolution than templates, so don't need to degrade the latter
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    # templates are 5 AA, we are 20 AA. 
    for i in range(disps.size):
        disp = disps[i]
        kernel = disp/(light*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1) # ready to be ndinterpolated?

def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=5200.,nfit=6.,bg='polynomial',nsrc=5,smooth='polynomial',bias=1e8,bias2=1e8):
    # Load in spectrum
    scispec = py.open('/data/ljo31b/EELs/esi/kinematics/J0837_spec.fits')[0].data
    varspec = py.open('/data/ljo31b/EELs/esi/kinematics/J0837_var.fits')[0].data
    sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/J0837_wl.fits')[0].data

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(9500.))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    sciwave = 10**sciwave[start:end]

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2

    # prepare the templates
    ntemps = len(templates)
    ntemp = 1
    result = []
    models = []
    t = []

    tmin = sciwave.min()
    tmax = sciwave.max()

    for template in templates:
        # Load the template
        tmpwave,tmpspec = np.loadtxt(template,unpack=True)
        tmpwave *= 10.
        tmpspec /= tmpspec.mean()

        # do we need to smooth these? Presumably not, because our resolution is better
        twave = np.log10(tmpwave)
        tmpscale = twave[1]-twave[0]
        t.append(getmodel(twave,tmpspec,tmpscale)) 
        print tmpscale
    return t
    '''if fit:
        result = finddispersion(scispec,varspec,t,tmpwave,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,nsrc=nsrc,smooth=smooth,bias=bias,bias2=bias2)
        return result
    elif read:
        result = readresults(scispec,varspec,t,tmpwave,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,nsrc=nsrc,smooth=smooth,bias=bias,bias2=bias2)
        return result
    else:
        return'''

name = 'bluepolyA10'
print name
result = run(0.4256,0.6411,fit=True,read=False,File = 'J0837_'+name,mask=np.array([[7580,7700],[6860,6900]]),nfit=5,bg='polynomial',lim=5200,nsrc=15,smooth='polynomial',bias=1e8)#,bias2=1e9)
#result = run(0.4246,0.6411,fit=False,read=True,File = 'J0837_'+name,mask=np.array([[7580,7700],[6860,6900]]),nfit=5,bg='polynomial',lim=5200,nsrc=15,smooth='polynomial',bias=1e8)#,bias2=1e9)

