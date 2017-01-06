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
    print q[f],pa[f]

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

OVRS=10
yo,xo = iT.overSample(np.zeros((200,200)).shape,OVRS)
yo-=99.5
xo-=99.5

S1 = fits[0][-3]*s1.pixeval(xo,yo,1./OVRS,csub=31)
S2 = fits[0][-2]*s2.pixeval(xo,yo,1./OVRS,csub=31)

Q,PA,ALPHA = np.median(q),np.median(pa),np.median(alpha) # for now - spline this later
xe = xo*cos(ALPHA) + yo*sin(ALPHA)
ye = xo*sin(ALPHA) - yo*cos(ALPHA)
r = (xe**2. * Q + ye**2. / Q)**0.5
'''pl.figure()
climshow(r)

pl.figure()
climshow(S1)'''
S3 = S1*2.*np.pi*r
S4 = S2*2.*np.pi*r
'''
pl.figure()
climshow(S3)
pl.show()

pl.figure()
pl.contour(xo,yo,S3,color='CornflowerBlue')
pl.figure()
pl.contour(xo,yo,S1,color='CornflowerBlue')
pl.show()'''

# cumsum S3 to get the half-light radius. We may need to oversample this.
argsort = np.argsort(r.flatten())
sortR = r.flatten()[argsort]
sortS = S3.flatten()[argsort]
pl.figure()
pl.plot(sortR,sortS)

# eval in new coords
r = np.logspace(-3,2.5,1501)
light = s1.eval(r)*fits[0][-2]*2.*np.pi*r
mod = splrep(r,light)
intlight = r*0.
for i in range(intlight.size):
    intlight[i] = splint(0,r[i],mod)


mod = splrep(intlight,r)
reff = splev(intlight.max()*0.5,mod)
print reff
