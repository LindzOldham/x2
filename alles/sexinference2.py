import numpy as np, pylab as pl, pyfits as py
from astLib import astCalc
import myEmcee_blobs as myEmcee
import pymc
from imageSim import SBObjects
from scipy.special import gammainc,gamma

vkidata = np.load('/data/ljo31/Lens/LensParams/VIK_phot_211_dict.npy')[()]
names = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()].keys()
radec = np.load('/data/ljo31/Lens/LensParams/radec.npy')[()]
sz = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
name = 'J0837'
cat = np.load('/data/ljo31b/EELs/sex/'+name+'_cat.npy')
v1,ev1,v2,ev2 = cat[:,5:9].T
i1,ei1,i2,ei2 = cat[:,27:31].T
ra,dec = cat[:,3:5].T
v,vi = v2,v2-i2
dvi = np.sqrt(ev2**2. + ei2**2.)
v_src,i_src,dv_src,di_src,vi_src,dvi_src, v_lens,i_lens,dv_lens,di_lens,vi_lens,dvi_lens, k_src, dk_src, k_lens, dk_lens = vkidata[name]
vi_src = v_src-i_src
ra_src,dec_src = radec[name]
R = np.sqrt((ra-ra_src)**2. + (dec-dec_src)**2.)*3600.

Re = pymc.Uniform('Re',0.011,1000)
n = pymc.Uniform('n',0.5,8)
mu = pymc.Uniform('mu',-1,4)
sigma = pymc.Uniform('sigma',0.1,2)
dx = pymc.Uniform('dx',-100,100)
dy = pymc.Uniform('dy',-100,100)
amp = pymc.Uniform('amp',0,1)
group = SBObjects.Sersic('group',{'x':dx,'y':dy,'pa':0,'q':1,'re':Re,'n':n})
cmd = SBObjects.Gauss('cmd',{'x':mu,'y':0,'pa':0,'q':1,'sigma':sigma})

pars = [Re,n,mu,sigma,dx,dy,amp]
cov = np.array([10,2,0.5,0.1,1,1,0.3])

# need to normalise the Sersic and the CMD over where we have dataz
@pymc.deterministic
def logL(value=0,p=pars):
    logp = 0
    re,n,mu,sigma,dx,dy,amp = [pars[i].value for i in range(len(pars))]
    b = 2.*n - 0.324
    sersic = b**(2.*n) / (2.*np.pi*n) / re**2. / gammainc(2.*n,b*(R.max()/re)**(1./n))/gamma(2.*n)
    sersic *= np.exp(-b*(R/re)**(1./n))
    gauss = np.exp(-0.5*(vi-mu)**2./sigma**2.) / np.sqrt(2.*np.pi*sigma**2.)
    prob = amp*sersic*gauss + (1.-amp)*np.ones(R.size)/(R.max()-R.min())/(vi.max()-vi.min())
    logp = np.log10(prob).sum()
    return logp

    
@pymc.observed
def likelihood(value=0.,lp=logL):
    return lp


S = myEmcee.Emcee(pars+[likelihood],cov=cov,nthreads=1,nwalkers=30)
S.sample(3000)
#outFile = '/data/ljo31/Lens/J1619/VKI_radgrad_'+str(X)
#f = open(outFile,'wb')
#cPickle.dump(S.result(),f,2)
#f.close()
result = S.result()
lp = result[0]
trace = np.array(result[1])
a1,a3 = np.unravel_index(lp.argmax(),lp.shape)
for i in range(len(pars)):
    pars[i].value = trace[a1,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)

pl.figure()
pl.plot(lp)
pl.show()
