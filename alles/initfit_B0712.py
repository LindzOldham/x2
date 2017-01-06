import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve,SBObjects
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl, numpy as np
import myEmcee_blobs as myEmcee
from scipy import optimize
from scipy.interpolate import RectBivariateSpline

## first thing: select brightest pixels in image plane and find a lens model that minimises their variance in the source plane.

dir = '/data/ljo31/Lens/B0712/'

image = py.open(dir+'B0712_nirc2_n_Kp_6x6.fits')[0].data
pl.imshow(image[200:-200,200:-200],interpolation='nearest',origin='lower')
pl.show()

points = np.array([[177.,84],[172.,70.],[96.,17.],[60.,131.]])
xpoints,ypoints = points.T

XL = pymc.Uniform('Lens x',80,120,value=100)
YL = pymc.Uniform('Lens y',80,120,value=100)
Q = pymc.Uniform('Lens q',0.1,1.,value=0.6)
PA = pymc.Uniform('Lens pa',0,180,value=135)
B = pymc.Uniform('Lens b',20,100,value=65)
#ETA = pymc.Uniform('Lens eta',0.5,1.5,value=1.)

#SH = pymc.Uniform('Shear',-0.05,0.05,value=0.) 
#SHPA = pymc.Uniform('Shear pa',0,180,value=135)

lens = MassModels.PowerLaw('lens',{'x':XL,'y':YL,'b':B,'eta':1.,'q':Q,'pa':PA})
#shear = MassModels.ExtShear('shear',{'x':XL,'y':YL,'b':SH,'pa':SHPA})
lenses = [lens]#,shear]

pars = [XL,YL,Q,PA,B]#,SH,SHPA]
cov = np.array([2.,2.,0.1,5.,2.])#,0.01,5.])

@pymc.deterministic
def logP(value=0.,p=pars):
    lp = 0.
    for lens in lenses:
        lens.setPars()
    xl,yl = pylens.getDeflections(lenses,[xpoints,ypoints])
    mx,my = np.mean(xl),np.mean(yl)
    rad = (xl-mx)**2. + (yl-my)**2.
    lp = -1*rad.sum()
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp


SS = AMAOpt(pars,[likelihood],[logP],cov=cov)
SS.sample(10000)
lp,trace,det = SS.result()
pl.figure()
pl.plot(lp)
pl.show() 
print 'results from optimisation:'
for i in range(len(pars)):
    pars[i].value = trace[-1,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)
    

#for key in det.keys():
#    pl.figure()
#    pl.plot(det[key])
#    pl.title(key)

pl.show()

for lens in lenses:
    lens.setPars()
xl,yl = pylens.getDeflections(lenses,[xpoints,ypoints])
pl.figure()
pl.scatter(xl,yl)
pl.show()

# this seems to be a stable solution! Now let's put it in the gui/emcee!
