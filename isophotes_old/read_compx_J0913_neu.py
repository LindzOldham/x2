import numpy as np, pylab as pl, pyfits as py
from linslens import EELsModels as L
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

result = np.load('/data/ljo31/Lens/J0913/twoband_212')#/LensModels/new/J0913_212')#
name='J0913'
from linslens import EELsModels as L
model = L.EELs(result,name)
model.Initialise()
print model.GetIntrinsicMags()
model.GetSourceSize(kpc=True)
z=model.z
Da=model.Da
scale=model.scale
srcs=model.srcs
imgs=model.imgs
fits=model.fits

models = model.models
comps = model.components
imgs = model.imgs

xc,yc = srcs[0].pars['x'],srcs[0].pars['y']
s1 = SBObjects.Sersic('s1',{'x':0.01,'y':0.01,'re':srcs[0].pars['re'],'n':srcs[0].pars['n'],'q':srcs[0].pars['q'],'pa':srcs[0].pars['pa']})
s2 = SBObjects.Sersic('s2',{'x':0.01,'y':0.01,'re':srcs[1].pars['re'],'n':srcs[1].pars['n'],'q':srcs[1].pars['q'],'pa':srcs[1].pars['pa']})

yo,xo = iT.overSample(np.zeros((81,81)).shape,1)
yo-=40.
xo-=40.

r = ((xo)**2. + (yo)**2.)**0.5

S1 = fits[0][-3]*s1.pixeval(xo,yo,1.,csub=31)
S2 = fits[0][-2]*s2.pixeval(xo,yo,1.,csub=31)
S3 = fits[0][-3]*s1.pixeval(xo,yo,1.,csub=31)*2.*np.pi*r
S4 = fits[0][-2]*s2.pixeval(xo,yo,1.,csub=31)*2.*np.pi*r

'''pl.figure()
climshow(S1)
pl.figure()
climshow(r)'''

fracs = [0.25,0.5,0.9]
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
 
    # now we want to fit each ellipse
    for i in range(len(y)):
        X,Y = x[i],y[i]
        a = pymc.Uniform('a',0,150-5*i)
        b = pymc.Uniform('b',0,150-5*i)
        alpha = pymc.Uniform('alpha',0.3,np.pi/2.)
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
        outFile = '/data/ljo31/Lens/compx_J0901_neu_'+str(fracs[i])
        f = open(outFile,'wb')
        cPickle.dump(S.result(),f,2)
        f.close()
        result = S.result()
        lp,trace,dic,_= result
        a2,a3 = np.unravel_index(lp.argmax(),lp.shape)
        for j in range(len(pars)):
            pars[j].value = trace[a2,a3,j]
            print "%18s  %8.3f"%(pars[j].__name__,pars[j].value)

    

#(method 3: don't cumsum, but just put knots down. Like setting up a spline). But this will be hard in the centre, where there aren't really any points!
frac2s = [0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99,0.999]

def ellipse2(xo,yo,S1,color='CornflowerBlue',fit=True):
    
    sortH = np.sort(S1.flatten())
    lvl00 = S1.max()
    
    def lvl(frac):
        return sortH[sortH>sortH.max()*frac].min()
    
    # extract contours
    pl.figure()
    cn = pl.contour(xo,yo,S1,[lvl(frac) for frac in frac2s],colors=color) 
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
    np.save('/data/ljo31/Lens/pylathon/XY_neu',[x,y])

    # now we want to fit each ellipse
    for i in range(len(y)):
        X,Y = x[i],y[i]
        a = pymc.Uniform('a',0,150-5*i)
        b = pymc.Uniform('b',0,150-5*i)
        alpha = pymc.Uniform('alpha',0.3,np.pi/2.)
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
        outFile = '/data/ljo31/Lens/compx_J0901_neu_'+str(fracs[i])
        f = open(outFile,'wb')
        cPickle.dump(S.result(),f,2)
        f.close()
        result = S.result()
        lp,trace,dic,_= result
        a2,a3 = np.unravel_index(lp.argmax(),lp.shape)
        for j in range(len(pars)):
            pars[j].value = trace[a2,a3,j]
            print "%18s  %8.3f"%(pars[j].__name__,pars[j].value)

ellipse(xo,yo,S1,fit=True)
#ellipse2(xo,yo,S1,fit=False)

# now we've fitted I(R) as a series of ellipses.
# Now we want to extract q, pa (could spline to get q(r), pa(r) or just assume it's constant for now)

files = glob.glob('/data/ljo31/Lens/compx_J0901_neu_*')
pa = np.zeros(len(files))
q,a,b,alpha = pa*0.,pa*0.,pa*0.,pa*0.

for f in range(len(files)):
    file = files[f]
    result = np.load(file)
    lp,trace,dic,_= result 
    ftrace = trace[trace.shape[0]*0.5:].reshape(trace.shape[0]*0.5*trace.shape[1],trace.shape[2])
    a[f],b[f],alpha[f] = np.percentile(ftrace,50,axis=0)
    q[f],pa[f] = b[f]/a[f], alpha[f]*180./np.pi
    print a[f],b[f],alpha[f],q[f],pa[f]

