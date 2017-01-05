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

# trying out the two-component run...

# now we've fitted I(R) as a series of ellipses.
# Now we want to extract q, pa (could spline to get q(r), pa(r) or just assume it's constant for now)

files = glob.glob('/data/ljo31/Lens/isophotes/compx_J0913_S1S2_*')
pa = np.zeros(len(files))
q,a,b,alpha,R = pa*0.,pa*0.,pa*0.,pa*0.,pa*0.

# make a spline model for q(r), pa(r)
X,Y,_ = np.load('/data/ljo31/Lens/isophotes/J0913_XYf.npy')

for f in range(len(files)):
    file = files[f]
    result = np.load(file)
    lp,trace,dic,_= result 
    ftrace = trace[trace.shape[0]*0.5:].reshape(trace.shape[0]*0.5*trace.shape[1],trace.shape[2])
    a[f],b[f],alpha[f] = np.percentile(ftrace,50,axis=0)
    q[f],pa[f] = b[f]/a[f], alpha[f]*180./np.pi
    print q[f],pa[f]
    xx,yy = X[f],Y[f]
    xxe = xx*cos(alpha[f]) + yy*sin(alpha[f])
    yye = xx*sin(alpha[f]) - yy*cos(alpha[f])
    rr = (xxe**2. * q[f] + yye**2. / q[f])**0.5
    R[f] = np.median(rr)

ii=np.argsort(R)
print np.logspace(np.min(R)+0.3,np.max(R)-0.3,5)
qmod = splrep(R[ii],q[ii],t=np.logspace(np.log10(np.min(R)+0.3),np.log10(np.max(R)-0.3),3))
#alphamod = splrep(R[ii],alpha[ii],t=np.logspace(np.min(R)+0.3,np.max(R)-0.3,5))
pl.figure()
pl.plot(R,q,'o')
pl.plot(R,splev(R,qmod))
pl.figure()
pl.plot(R,alpha)
pl.show()

# define radial coordinates so we can integrate surface brightness profile to get luminosity
result = np.load('/data/ljo31/Lens/J0913/twoband_212')#/LensModels/new/J0913_212')#
name='J0913'
from linslens import EELsModels as L
model = L.EELs(result,name)
model.Initialise()
model.GetIntrinsicMags()
model.GetSourceSize(kpc=True)
z,Da,scale=model.z,model.Da,model.scale
srcs=model.srcs
fits=model.fits

xc,yc = srcs[0].pars['x'],srcs[0].pars['y']
s1 = SBObjects.Sersic('s1',{'x':0.0,'y':0.0,'re':srcs[0].pars['re'],'n':srcs[0].pars['n'],'q':srcs[0].pars['q'],'pa':srcs[0].pars['pa']})
s2 = SBObjects.Sersic('s2',{'x':0.0,'y':0.0,'re':srcs[1].pars['re'],'n':srcs[1].pars['n'],'q':srcs[1].pars['q'],'pa':srcs[1].pars['pa']})

## re-evaluate on a smaller grid to do the integral
yo=np.logspace(-2,2.5,100)
xo=np.logspace(-2,2.5,100)
xo,yo = np.meshgrid(xo,yo)

Q,PA,ALPHA = np.median(q),np.median(pa),np.median(alpha) # for now - spline this later



xe = xo*cos(ALPHA) + yo*sin(ALPHA)
ye = xo*sin(ALPHA) - yo*cos(ALPHA)
r = (xe**2. * Q + ye**2. / Q)**0.5
OVRS=1.
S1 = fits[0][-3]*s1.pixeval(xo,yo,1./OVRS,csub=31)
print S1.shape, r.shape
S2 = fits[0][-2]*s2.pixeval(xo,yo,1./OVRS,csub=31)
S3 = S1*2.*np.pi*r
S4 = S2*2.*np.pi*r

argsort = np.argsort(r.flatten())
sortR = r.flatten()[argsort]
sortS = (S3.flatten()+S4.flatten())[argsort]
pl.figure()
pl.plot(sortR,sortS)
pl.show()

mod = splrep(sortR,sortS,t=np.logspace(-1.8,2.3,50))
pl.figure()
pl.plot(sortR,sortS,'o')
pl.plot(sortR,splev(sortR,mod))
pl.show()

intlight = sortR*0.
for i in range(intlight.size):
    intlight[i] = splint(0,sortR[i],mod)

pl.figure()
pl.plot(sortR,intlight/np.sum(intlight),'.')
pl.axhline(intlight[-1]*0.5/np.sum(intlight))
pl.show()

mod = splrep(intlight,sortR)
reff = splev(intlight[-1]*0.5,mod)
print reff, reff*0.05*scale
print model.Re_v/0.05
print model.Re_v*scale
