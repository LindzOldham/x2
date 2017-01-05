import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from fitter import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g

dir = '/data/ljo31b/EELs/esi/INDOUS/'
templates = ['102328_K3III.fits','163588_K2III.fits','107950_G5III.fits','124897_K1III.fits','168723_K0III.fits','111812_G0III.fits','148387_G8III.fits','188350_A0III.fits','115604_F2III.fits']

for i in range(len(templates)):
    templates[i] = dir+templates[i]

twave  = np.log10(T.wavelength(templates[0],0))
tscale = twave[1]-twave[0]

VGRID = 1.
light = 299792.458
ln10 = np.log(10.)

# science data resolution, template resolution
sigtmp1 =  0.44/7000. * light
sigtmp2 = light/500.

def getmodel4(twave,tspec,tscale,smin=5.,smax=501):
    match = tspec.copy()
    kernel = (sigsci**2.+50.**2-sigtmp1**2)**0.5/(299792.*ln10*tscale)
    match = ndimage.gaussian_filter1d(tspec.copy(),kernel)
    disps = np.arange(smin,smax,VGRID)
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        disp = (disp**2.-50.**2.)**0.5
        if np.isnan(disp)==True:
            disp = 5.
        kernel = disp/(light*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1)


def getmodel(twave,tspec,tscale,smin=5.,smax=501):
    match = tspec.copy()
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        dispkern = (disp**2.+sigsci**2.-sigtmp1**2.)**0.5
        if np.isnan(dispkern)==True:
            dispkern = 5.
        kernel = dispkern/(light*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1)

def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=6000.,nfit=6.,bg='polynomial',bias=15.):
    # Load in spectrum
    scispec = py.open('/data/ljo31b/EELs/esi/spectra/'+name[:5]+'_spec.fits')[0].data
    varspec = py.open('/data/ljo31b/EELs/esi/spectra/'+name[:5]+'_var.fits')[0].data
    sciwave = py.open('/data/ljo31b/EELs/esi/spectra/'+name[:5]+'_wl.fits')[0].data

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
    ntemps = len(templates)
    ntemp = 1
    result = []
    models = []
    t = []

    tmin = sciwave.min()
    tmax = sciwave.max()

    for template in templates:
        # Load the template
        file = py.open(template)
        tmpspec = file[0].data.astype(np.float64)
        tmpwave = T.wavelength(template,0)
        tmpspec /= tmpspec.mean()

        # do we need to smooth these? Presumably not, because our resolution is better
        twave = np.log10(tmpwave)
        tmpscale = twave[1]-twave[0]
        t.append(getmodel(twave,tmpspec,tmpscale)) 
    if fit:
        result = finddispersion(scispec,varspec,t,tmpwave,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,bias=bias)
        return result
    elif read:
        result = readresults(scispec,varspec,t,tmpwave,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,bias=bias)
        return result
    else:
        return

'''
name = 'J0901_0'
print name
result = run(0.311,0.586,fit=True,read=False,File = name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5800,bias=1e8)#,restmask=np.array([[3900,3990]]))
result = run(0.311,0.586,fit=False,read=True,File = name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5800,bias=1e8)#,restmask=np.array([[3900,3990]]))
pl.title(name[:5])
pl.show()
'''
'''
name = 'J0913_0'
print name
result = run(0.395,0.539,fit=True,read=False,File = name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5700,bias=1e8)#,restmask=np.array([[3900,3990]]))
result = run(0.395,0.539,fit=False,read=True,File = name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5700,bias=1e8)#,restmask=np.array([[3900,3990]]))
pl.title(name[:5])
pl.show()
'''



name='J1248'
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]
resolutions = np.load('/data/ljo31b/EELs/esi/spectra/resolutions_from_skylines.npy')[()]
sigsci = resolutions[name[:5]] * light
print sigsci

result = run(lz[name],sz[name],fit=True,read=False,File = name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5500.,bias=1e8)
result = run(lz[name],sz[name],fit=False,read=True,File = name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5500.,bias=1e8)
pl.title(name[:5])
pl.show()



#from tools import gus_plotting as g
'''lp,trace,dic,_ = result
g.changechain(trace,filename='/data/ljo31b/EELs/esi/kinematics/inference/'+name+'_chain')
chain = np.loadtxt('/data/ljo31b/EELs/esi/kinematics/inference/'+name+'_chain')
np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/'+name+'_chain.txt',chain[5000:,1:])
g.triangle_plot(chain,burnin=125,axis_labels=['$v_l$',r'$\sigma_l$','$v_s$',r'$\sigma_s$'])
'''


