import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from stitchfitter3_restframe2 import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g
from scipy.interpolate import splrep, splev
from scipy import sparse

chabrier = np.load('/home/mauger/python/stellarpop/chabrier.dat')
ages = chabrier['age']
wave = chabrier['wave']
spectra = chabrier[6]
c = (wave>3000.)&(wave<8000.)
wave=wave[c]

# take 2 gyr, 5 gyr, 6 gyr, 7 gyr, 9 gyr?
idx = np.where((ages==5e9)|(ages==2e9)|(ages==6e9)|(ages==7e9)|(abs(ages-9e9)<0.1e9)|(abs(ages-8e9)<0.1e9)|(abs(ages-7.5e9)<0.1e9))
spex = [spectra[ii][c]/spectra[ii][c].mean() for ii in idx[0]]

light = 299792.458
ln10 = np.log(10.)
sigsci = 20.40

'''pixscale = 1.
tempSigma = light*3./wave/2.355
X = np.arange(wave.size)
disps = np.arange(130.,500.,1.)
ogrids = [np.empty((disps.size,wave.size)) for s in spex]
for i in range(disps.size):
    kernel = wave*(disps[i]**2. + sigsci**2. - tempSigma**2.)**0.5/light
    kpix = kernel/pixscale
    kpix2 = kpix**2.
    pmax = int(kpix.max()*4+1)
    rcol = np.linspace(-pmax,pmax,2*pmax+1).repeat(X.size).reshape((2*pmax+1,X.size))
    col = rcol+X
    row = X.repeat(2*pmax+1).reshape((X.size,2*pmax+1)).T
    c = (col>=0)&(col<wave.size)&(abs(rcol)<4*kpix[row])
    col = col[c]
    row = row[c]
    rcol = rcol[c]
    kpix2 = kpix2[row]
    wht = np.exp(-0.5*rcol**2/kpix2)/(2*np.pi*kpix2)**0.5
    M = sparse.coo_matrix((wht,(row,col)),shape=(X.size,X.size))

    for j in range(len(spex)):
        s = spex[j].copy()
        s = M*s
        ogrids[j][i] = s

X = disps.tolist()
tx = np.array([X[0]]+X+[X[-1]])
Y = np.log10(wave).tolist()
ty = np.array([Y[0]]+Y+[Y[-1]])
model = [(tx,ty,ogrid.flatten(),1,1) for ogrid in ogrids]
np.save('/data/ljo31b/EELs/esi/BC03models_EELs',model)'''

dir = '/data/ljo31b/EELs/esi/PICKLES/'
templates2 = ['K3III.dat','K2III.dat','G5III.dat','K1III.dat','K0III.dat','G0III.dat','G8III.dat','A0III.dat','F2III.dat']

for i in range(len(templates2)):
    templates2[i] = dir+templates2[i]

sigtmp2 = light/500./2.355

def getmodel(twave,tspec,tscale,sigtmp,smin=5.,smax=501):
    match = tspec.copy()
    disps = np.arange(smin,smax,1.)
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


def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=5000.,nfit=6.,bg='polynomial',bias=1e8,restmask=None,srclim=6000.,lenslim=5500.,wid=1.,typ=None):
    # Load in spectrum
    if typ=='lens':
        print 'lens'
        try:
            scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_spec_lens.fits')[0].data
            varspec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_var_lens.fits')[0].data
            sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_wl_lens.fits')[0].data
        except:
            scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'0_spec_lens.fits')[0].data
            varspec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'0_var_lens.fits')[0].data
            sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'0_wl_lens.fits')[0].data
    else:
        print 'source'
        try:
            scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_spec_source.fits')[0].data
            varspec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_var_source.fits')[0].data
            sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'_wl_source.fits')[0].data
        except:
            scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'0_spec_source.fits')[0].data
            varspec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'0_var_source.fits')[0].data
            sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/'+name+'_ap_'+str(wid)+'0_wl_source.fits')[0].data
    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(8500.))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]
    
    # convert esi air wavelengths to vacuum wavelengths
    # sdss website, cf, morton 1991
    vac = np.linspace(3800,11000,(11000-3800+1)*5)
    air = vac / (1. + 2.735182e-4 + (131.4182/vac**2.) + (2.76249e8/vac**8))
    model = splrep(air,vac)
    sciwave = splev(sciwave,model)
    # see what difference this makes

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2

    # prepare the templates
    ntemps1 = len(idx[0])
    ntemps2 = len(templates2)

    result = []
    models = []
    t1,t2 = [],[]
    
    t1 = np.load('/data/ljo31b/EELs/esi/BC03models_EELs.npy')    
    tmpwave1 = wave.copy()

    for template in templates2:
        tmpwave2,tmpspec2 = np.loadtxt(template,unpack=True)
        tmpwave2 *= 10.
        tmpspec2 /= tmpspec2.mean()
        
        twave2 = np.log10(tmpwave2)
        tmpscale2 = twave2[455]-twave2[454] # also in region of interest
        t2.append(getmodel(twave2,tmpspec2,tmpscale2,sigtmp2)) 

    ntemps1,ntemps2 = len(t1), len(t2)

    if fit:
        result = finddispersion(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,lenslim=lenslim,srclim=srclim)
        return result
    elif read:
        result = readresults(scispec,varspec,t1,t2,tmpwave1,tmpwave2,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,restmask=restmask,lenslim=lenslim,srclim=srclim)
        return result
    else:
        return


sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
lz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]

import sys
name,wid,typ = sys.argv[1],sys.argv[2],sys.argv[3]
print name,wid,typ
triangle=False
print typ
if name=='J0837':
    filename = name+'_'+wid+'_'+typ+'_vac_bc03'
    result = run(0.4246,0.6411,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5295,5325],[5570,5585],[6860,6900]]),nfit=6,bg='polynomial',lim=4900,srclim=5300,lenslim=5000,bias=1e8,typ=typ,wid=wid)
    result = run(0.4246,0.6411,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5295,5325],[5570,5585],[6860,6900]]),nfit=6,bg='polynomial',lim=4900,srclim=5300,lenslim=5000,bias=1e8,typ=typ,wid=wid)
    
    pl.suptitle(name)
    pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/apertures/final/'+filename+'.pdf')
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/J0837esibc03_seg1.pdf')
    pl.show()

elif name == 'J0901':
    filename = name+'_'+wid+'_'+typ+'_esi_bc03'
    result = run(0.311,0.586,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900,bias=1e8,srclim=5150.,lenslim=4950.,typ=typ,wid=wid)
    result = run(0.311,0.586,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=12,bg='polynomial',lim=4900,bias=1e8,srclim=5150.,lenslim=4950.,typ=typ,wid=wid)
    
    pl.suptitle(name)
    pl.savefig('/data/ljo31b/EELs/esi/kinematics/inference/plots/'+filename+'.pdf')
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/J0901esibc03.pdf')
    pl.show()

elif name == 'J0913':
    sigsci = resolutions[name] * light
    print sigsci
    filename = name+'_'+wid+'_'+typ+'_vac_bc03'
    result = run(0.395,0.539,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=4500,bias=1e8,srclim=5700.,lenslim=5150.,typ=typ,wid=wid)
    result = run(0.395,0.539,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=4500,bias=1e8,srclim=5700.,lenslim=5150.,typ=typ,wid=wid)
    pl.suptitle(name)
    pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/apertures/final/'+filename+'.pdf')
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/J0913esibc03.pdf')
    pl.show()

elif name == 'J1125':
    sigsci = resolutions[name] * light
    print sigsci
    filename = name+'_'+wid+'_'+typ +'_vac_bc03'
    result = run(0.442,0.689,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=4500.,bias=1e8,srclim=6200.,lenslim=5300.,typ=typ,wid=wid)
    result = run(0.442,0.689,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=4500.,bias=1e8,srclim=6200.,lenslim=5300.,typ=typ,wid=wid)
    pl.suptitle(name[:5])
    pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/apertures/final/'+filename+'.pdf')
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/texplots/J1125esibc03.pdf')
    pl.show()

elif name == 'J1144':
    sigsci = resolutions[name] * light
    print sigsci
    filename = name+'_'+wid+'_'+typ+'_vac'
    result = run(0.372,0.706,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=5000.,bias=1e8,srclim=6300.,lenslim=5050.,typ=typ,wid=wid)
    result = run(0.372,0.706,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=5000.,bias=1e8,srclim=6300.,lenslim=5050.,typ=typ,wid=wid)
    #pl.suptitle(name[:5])
    #pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/apertures/final/'+filename+'.pdf')
    #pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+filename+'.png')
    pl.show()

elif name == 'J1218':
    sigsci = resolutions[name] * light
    print sigsci
    filename = name+'_'+wid+'_'+typ+'_vac'
    result = run(0.3182,0.6009,fit=True,read=False,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e8,srclim=5900.,lenslim=4900.,typ=typ,wid=wid)
    result = run(0.3182,0.6009,fit=False,read=True,File = filename,mask=np.array([[7580,7700],[5570,5585],[6860,6900]]),nfit=10,bg='polynomial',lim=4750.,bias=1e8,srclim=5900.,lenslim=4900.,typ=typ,wid=wid)
    pl.suptitle(name[:5])
    pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/apertures/final/'+filename+'.pdf')
    pl.savefig('/data/ljo31/public_html/Lens/sdss_eels_spectra/'+filename+'.png')
    pl.show()


'''
# J1248
name = 'J1248_wide'
resolutions = np.load('/data/ljo31b/EELs/esi/spectra/resolutions_from_skylines.npy')[()]
sigsci = resolutions[name[:5]] * light
print name, sigsci
result = run(0.304,0.528,fit=True,read=False,File = name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4750.,bias=1e8,srclim=5900.,lenslim=4900.)
result = run(0.304,0.528,fit=False,read=True,File = name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=4750.,bias=1e8,srclim=5900.,lenslim=4900.)
pl.title(name[:5])
#pl.savefig('/data/ljo31b/EELs/esi/kinematics/plots/'+name[:5]+'.pdf')
'''


'''lp,trace,dic,_ = result
chain = g.changechain(trace,filename='/data/ljo31b/EELs/esi/kinematics/inference/'+name+'_chain')
np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/'+name+'_chain.txt',chain[6000:,1:])
g.triangle_plot(chain,burnin=125,axis_labels=['$v_l$',r'$\sigma_l$','$v_s$',r'$\sigma_s$'])
pl.show()
'''



'''
to print results:
result = np.load('/data/ljo31b/EELs/esi/kinematics/inference/J0901')
zl=lzs['J0901']
zs=szs['J0901']

lp,trace,dic,_ = result
medians = []
diffs = []

for key in dic.keys():
    f = dic[key][150:,:]                                          
    f = f.flatten()
    l,m,u = np.percentile(f,16,axis=0),np.percentile(f,50,axis=0),np.percentile(f,84,axis=0)
    med, dmed = m, np.mean((m-l,u-m))
    medians.append(med)
    diffs.append(dmed)

for i in range(len(dic.keys())):
    print dic.keys()[i], '%.2f'%medians[i], '\pm', '%.2f'%diffs[i]

# update redshifts
print zl+medians[0]/light
print zs+medians[-1]/light

'''
