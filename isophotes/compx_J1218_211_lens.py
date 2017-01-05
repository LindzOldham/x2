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

name='J1218'
ls,num = 'lens','211'
plot=True
result = np.load('/data/ljo31/Lens/'+name+'/twoband_0')#/LensModels/new/J0913_212')#
model = L.EELs(result,name)
model.Initialise()
print model.GalaxyObsMag()
model.GalaxySize(kpc=True)
z=model.z
Da=model.Da
scale=model.scale
gals=model.gals
fits=model.fits


xc,yc = gals[0].pars['x'],gals[0].pars['y']
dx,dy = gals[1].pars['x']-xc,gals[1].pars['y']-yc
print dx,dy
s1 = SBObjects.Sersic('s1',{'x':0.0,'y':0.0,'re':gals[0].pars['re'],'n':gals[0].pars['n'],'q':gals[0].pars['q'],'pa':gals[0].pars['pa']})
s2 = SBObjects.Sersic('s2',{'x':dx,'y':dy,'re':gals[1].pars['re'],'n':gals[1].pars['n'],'q':gals[1].pars['q'],'pa':gals[1].pars['pa']})

yo,xo = iT.overSample(np.zeros((200,200)).shape,1)
yo-=99.9
xo-=99.9

r = ((xo)**2. + (yo)**2.)**0.5

S1 = fits[0][0]*s1.pixeval(xo,yo,1.,csub=31)
S2 = fits[0][1]*s2.pixeval(xo,yo,1.,csub=31)

fracs = [0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]#,0.99,0.999]

# get q, pa by fitting SB isophotes
def ellipse(xo,yo,H,color='CornflowerBlue',fit=True):
    sortH = np.sort(H.flatten())
    cumH = sortH.cumsum()
    lvl00 = sortH.max()
    def lvl(frac):
        return sortH[cumH>cumH.max()*frac].min()
    
    # extract contours
    pl.figure()
    cn = pl.contour(xo,yo,H,[lvl(frac) for frac in fracs],colors=color)
    pl.axhline(0)
    pl.axvline(0)
    pl.show()
    
    if fit is False:
        return

    p,v,x,y = [],[],[],[]

    for i in range(len(fracs)):
        p.append(cn.collections[i].get_paths()[0])
        v.append(p[-1].vertices)
        X,Y = v[-1].T
        x.append(X)
        y.append(Y)
    np.save('/data/ljo31/Lens/isophotes/'+name+'_'+ls+'_'+num+'_XYf',[x,y,fracs])

    # now we want to fit each ellipse
    for i in range(len(y)):
        X,Y = x[i],y[i]
        a = pymc.Uniform('a',0,150-5*i)
        b = pymc.Uniform('b',0,150-5*i)
        alpha = pymc.Uniform('alpha',-0.5,np.pi/2.-0.55)
        pars = [a,b,alpha]
        cov=np.array([1.,1.,0.2])
        
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
        outFile = '/data/ljo31/Lens/isophotes/compx_'+name+'_'+ls+'_'+num+'_'+str(fracs[i])
        f = open(outFile,'wb')
        cPickle.dump(S.result(),f,2)
        f.close()
        result = S.result()
        lp,trace,dic,_= result
        a2,a3 = np.unravel_index(lp.argmax(),lp.shape)
        for j in range(len(pars)):
            pars[j].value = trace[a2,a3,j]
            print "%18s  %8.3f"%(pars[j].__name__,pars[j].value)

    if plot:
        result = np.load('/data/ljo31/Lens/isophotes/compx_'+name+'_'+ls+'_'+num+'_'+str(fracs[1]))
        lp,trace,dic,_ = result
        pl.figure()
        pl.subplot(211)
        pl.plot(dic['alpha'])
        pl.subplot(212)
        pl.plot(dic['a'])
        pl.show()

ellipse(xo,yo,S1+S2,fit=True)

# now we've fitted I(R) as a series of ellipses.
# Now we want to extract q, pa (could spline to get q(r), pa(r) or just assume it's constant for now)


# go to READ_....py!
