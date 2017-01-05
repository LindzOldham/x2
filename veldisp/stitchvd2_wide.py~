import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter2 import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g

dir = '/data/ljo31b/EELs/esi/INDOUS/' 
templates1 = ['102328_K3III.fits','163588_K2III.fits','107950_G5III.fits','124897_K1III.fits','168723_K0III.fits','111812_G0III.fits','148387_G8III.fits','188350_A0III.fits','115604_F2III.fits']

for i in range(len(templates1)):
    templates1[i] = dir+templates1[i]

dir = '/data/ljo31b/EELs/esi/PICKLES/'
templates2 = ['K3III.dat','K2III.dat','G5III.dat','K1III.dat','K0III.dat','G0III.dat','G8III.dat','A0III.dat','F2III.dat']

for i in range(len(templates2)):
    templates2[i] = dir+templates2[i]

VGRID = 1.
light = 299792.458
ln10 = np.log(10.)

# science data resolution, template resolution
sigsci = 6.78e-5 * light
sigtmp1 =  0.44/7000. * light
sigtmp2 = light/500.

def getmodel(twave,tspec,tscale,sigsci,sigtmp,smin=5.,smax=501):
    match = tspec.copy()
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        dispkern = (disp**2.+sigsci**2.-sigtmp**2.)**0.5
        if np.isnan(dispkern)==True:
            dispkern = 5. 
        kernel = dispkern/(light*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1)


def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=5000.,nfit=6.,bg='polynomial',bias=1e8,restmask=None):
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
    ntemps1 = len(templates1)
    ntemps2 = len(templates2)

    result = []
    models = []
    t1,t2 = [],[]
    
    for template in templates1:
        file = py.open(template)
        tmpspec1 = file[0].data.astype(np.float64)
        tmpwave1 = T.wavelength(template,0)
        tmpspec1 /= tmpspec1.mean()

        twave1 = np.log10(tmpwave1)
        tmpscale1 = twave1[1]-twave1[0]
        t1.append(getmodel(twave1,tmpspec1,tmpscale1,sigsci,sigtmp1)) 
    
    for template in templates2:
        tmpwave2,tmpspec2 = np.loadtxt(template,unpack=True)
        tmpwave2 *= 10.
        tmpspec2 /= tmpspec2.mean()

        twave2 = np.log10(tmpwave2)
        tmpscale2 = twave2[1]-twave2[0]
        t2.append(getmodel(twave2,tmpspec2,tmpscale2,sigsci,sigtmp2)) 

    ntemps1,ntemps2 = len(t1), len(t2)

    if fit:
        result = finddispersion(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,restmask=restmask)
        return result
    elif read:
        result = readresults(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,restmask=restmask)
        return result
    else:
        return

name = '3' # 10,000
print name
result = run(0.4256,0.6411,fit=True,read=False,File = 'J0837_'+name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5000,bias=1e8)#,restmask=np.array([[3900,3990]]))
result = run(0.4246,0.6411,fit=False,read=True,File = 'J0837_'+name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5000,bias=1e8)#,restmask=np.array([[3900,3990]]))

lp,trace,dic,_ = result
chain = g.changechain(trace,filename='/data/ljo31b/EELs/esi/kinematics/inference/J0837_'+name+'_chain')
np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/J0837_'+name+'_chain.txt',chain[5000:,1:])
g.triangle_plot(chain,burnin=125,axis_labels=['$v_l$',r'$\sigma_l$','$v_s$',r'$\sigma_s$'])
