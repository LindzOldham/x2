import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g

t1 = np.load('/data/ljo31b/EELs/esi/INDOUS/interpolators.npy')
t2 = np.load('/data/ljo31b/EELs/esi/KURUCZ/interpolators.npy')
scale1, scale2 = 2.5e-5,0.00229785976228

file1 = '/data/ljo31b/EELs/esi/INDOUS/102328_K3III.fits'
file2 = '/data/ljo31b/EELs/esi/KURUCZ/K0III.dat'

twave1  = T.wavelength(file1,0)
twave2 = np.loadtxt(file2,unpack=True,usecols=(0,))

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
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2

    # prepare the templates
    ntemps1,ntemps2 = len(t1), len(t2)

    if fit:
        result = finddispersion(scispec,varspec,t1,t2,twave1,twave2,scale1,scale2,datascale,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg)
        return result
    elif read:
        result = readresults(scispec,varspec,t1,t2,twave1,twave2,scale1,scale2,datascale,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg)
        return result
    else:
        return

name = 'kpoly2'
print name
#result = run(0.4256,0.6411,fit=True,read=False,File = 'J0837_'+name,mask=np.array([[7580,7700],[6860,6900]]),nfit=5,bg='polynomial',lim=5200,bias=1e8)
result = run(0.4246,0.6411,fit=False,read=True,File = 'J0837_'+name,mask=np.array([[7580,7700],[6860,6900]]),nfit=5,bg='polynomial',lim=5200,bias=1e8)

#lp,trace,dic,_ = result
#chain = g.changechain(trace,filename='/data/ljo31b/EELs/esi/kinematics/inference/J0837_'+name+'_chain')
#np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/J0837_'+name+'_chain.txt',chain[5000:,1:])
#g.triangle_plot(chain,burnin=125,axis_labels=['$v_l$',r'$\sigma_l$','$v_s$',r'$\sigma_s$'])
