import numpy as np, pylab as pl, pyfits as py
from linslens import EELsLensModels as L
from tools.simple import printn, climshow
from scipy.interpolate import splrep, splev,splint
import indexTricks as iT
from scipy import ndimage
from tools.fitEllipse import *
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from numpy import cos, sin
import glob
from tools.simple import *
from imageSim import SBModels,convolve,SBObjects
from astLib import astCalc

fracs = [0.21,0.05,0.003]

# get q, pa by fitting SB isophotes
def ellipse(xo,yo,color='CornflowerBlue',fit=True):
    xbins=np.linspace(min(xo)-0.02,max(xo)+0.02,10)
    ybins=np.linspace(min(yo)-0.1,max(yo)+0.1,10)
    H,xbins,ybins = np.histogram2d(xo,yo,bins=[xbins,ybins])
    
    sortH = np.sort(H.flatten())
    cumH = sortH.cumsum()
    lvl00 = sortH.max()
    def lvl(frac):
        return sortH[cumH>cumH.max()*frac].min()
    
    # extract contours
    xbins = [xbins[i]+0.5*(xbins[i+1]-xbins[i]) for i in range(len(xbins)-1)]
    ybins = [ybins[i]+0.5*(ybins[i+1]-ybins[i]) for i in range(len(ybins)-1)]
    pl.figure()
    cn = pl.contour(xbins,ybins,H,[lvl(frac) for frac in fracs],colors=color)
    pl.show()
    
    if fit is False:
        return

    p = cn.collections[-1].get_paths()[0]
    v = p.vertices
    X,Y = v.T

    a = pymc.Uniform('a',0,1,value=0.1)
    b = pymc.Uniform('b',0,0.1,value=0.01)
    alpha = pymc.Uniform('alpha',0.3,np.pi/2.)
    pars = [a,b,alpha]
    cov=np.array([0.05,0.005,0.2])
        
    @pymc.deterministic
    def logP(value=0.,p=pars):
        A = (X*cos(alpha.value) + Y*sin(alpha.value))/a.value
        B = (X*sin(alpha.value) - Y*cos(alpha.value))/b.value
        eq = A**2. + B**2. - 1.
        return np.sum(-eq**2.)
                
    @pymc.observed
    def likelihood(value=0.,lp=logP):
        return lp

    S = myEmcee.Emcee(pars+[likelihood],cov=cov,nthreads=1,nwalkers=100)
    S.sample(2000)
    outFile = '/data/ljo31/Lens/ellipse_hugemodels/huge_211_mu_re_lensgals_'+str(name)
    f = open(outFile,'wb')
    cPickle.dump(S.result(),f,2)
    f.close()
    result = S.result()
    lp,trace,dic,_= result
    a2,a3 = np.unravel_index(lp.argmax(),lp.shape)
    for j in range(len(pars)):
        pars[j].value = trace[a2,a3,j]
        print "%18s  %8.3f"%(pars[j].__name__,pars[j].value)

    # plot model
    R = np.arange(0,2.*np.pi, 0.01)
    xx = a.value*cos(R)*cos(alpha.value) - b.value*sin(R)*sin(alpha.value)
    yy = a.value*cos(R)*sin(alpha.value) + b.value*sin(R)*cos(alpha.value)
    pl.plot(X,Y,'b')
    pl.plot(xx,yy, color = 'red')
    pl.show()
    pl.figure()
    pl.subplot(311)
    pl.plot(dic['a'])
    pl.subplot(312)
    pl.plot(dic['b'])
    pl.subplot(313)
    pl.plot(dic['alpha'])
    pl.show()

    # extract rho
    sx2 = cos(alpha.value)**2. / a.value**2. + sin(alpha.value)**2. / b.value**2.
    sy2 = cos(alpha.value)**2. / b.value**2. + sin(alpha.value)**2. / a.value**2.
    sxy = -1.*cos(alpha.value)*sin(alpha.value)*(1./a.value**2. - 1./b.value**2.)
    print 'rho', sxy/(sx2*sy2)**0.5
    RHO = sxy/(sx2*sy2)**0.5
    return RHO

'''sx2 = cos(alpha)**2. / a**2. + sin(alpha)**2. / b**2.
sy2 = cos(alpha)**2. / b**2. + sin(alpha)**2. / a**2.
sxy = -1.*cos(alpha)*sin(alpha)*(1./a**2. - 1./b**2.)'''
    
    

lz = np.load('/data/ljo31/Lens/LensParams/LensRedshiftsUpdated.npy')[()]
names = py.open('/data/ljo31/Lens/LensParams/Phot_1src_new.fits')[1].data['name']

rho = np.zeros(len(names))

for kk in range(len(names)):
    name=names[kk]
    print name
    re,mag = np.load('/data/ljo31/Lens/Analysis/ReMagPDFs_twoband_lensgals_'+str(name)+'.npy')
    rev,rei = re.T
    magv,magi = mag.T
    z = lz[name][0]
    Da = astCalc.da(z)
    scale = Da*1e3*np.pi/180./3600.
    re_arcsec = rev/scale
    muv = magv + 2.5*np.log10(2.*np.pi*re_arcsec**2.)
    x,y = np.log10(rev), muv
    x,y=x-np.median(x), y-np.median(y)
    rho[kk] = ellipse(x,y,fit=True)
    print rho[kk]
    #K = ellipse(x,y,fit=True)
    #print K

np.save('/data/ljo31/Lens/LensParams/rho_huge_211_lensgals',rho)

