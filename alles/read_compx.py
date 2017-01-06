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
yo,xo = iT.overSample(imgs[0].shape,1)
S1 = fits[0][-3]*srcs[0].pixeval(xo,yo,1.,csub=31)
S2 = fits[0][-2]*srcs[1].pixeval(xo,yo,1.,csub=31)

files = glob.glob('/data/ljo31/Lens/compx_J0901_*')
a,b,alpha = np.ones(len(files)),np.ones(len(files)),np.ones(len(files))
pl.figure()
for i in range(len(files)):
    file=files[i]
    result = np.load(file)
    lp,trace,dic,_=result
    ftrace = trace[trace.shape[0]*0.5:].reshape(trace.shape[0]*0.5*trace.shape[1],trace.shape[2])
    a[i],b[i],alpha[i] = np.percentile(ftrace,50,axis=0)
    print a[i],b[i],alpha[i]

    R = np.arange(0,2.*np.pi, 0.01)
    xx = a[i]*cos(R)*cos(alpha[i]) - b[i]*sin(R)*sin(alpha[i])
    yy = a[i]*cos(R)*sin(alpha[i]) + b[i]*sin(R)*cos(alpha[i])
    pl.plot(xx,yy, color = 'SteelBlue')
    

# containing 50 percent of the light
result = np.load('/data/ljo31/Lens/compx_J0901_0.5')
lp,trace,dic,_= result 
ftrace = trace[trace.shape[0]*0.5:].reshape(trace.shape[0]*0.5*trace.shape[1],trace.shape[2])
a,b,alpha = np.percentile(ftrace,50,axis=0)
q,pa = b/a, alpha*180./np.pi

r = (a*b)**0.5 * 0.05 * scale
print r
