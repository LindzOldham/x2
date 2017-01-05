import numpy,pyfits,pylab
from imageSim import SBObjects,convolve
from pylens import MassModels,pylens
from scipy import optimize
import pymc
from SampleOpt import AMAOpt


img = pyfits.open('img.fits')[0].data.copy()
sig = pyfits.open('sig.fits')[0].data.copy()
psf = pyfits.open('psf.fits')[0].data.copy()

psf /= psf.sum()

sig[64:75,121:130] = sig.max()*1e3

y,x = numpy.indices(img.shape).astype(numpy.float32)

x1,x2 = 55,103
y1,y2 = 60,100
x1,x2 = 22,133
y1,y2 = 35,115
x1,x2 = 50,100
y1,y2 = 57,107
#x1,x2 = 45,113
#y1,y2 = 50,110
#x1,x2 = 1,151
#y1,y2 = 1,161

if 1==2:
    img = img[y1:y2,x1:x2].copy()
    sig = sig[y1:y2,x1:x2].copy()
    psf = convolve.convolve(img,psf)[1]
    x = x[y1:y2,x1:x2].copy()
    y = y[y1:y2,x1:x2].copy()
else:
    psf = convolve.convolve(img,psf)[1]


img += 1.

rhs = (img/sig).ravel()
sflt = sig.ravel()
xflt = x.ravel()
yflt = y.ravel()

2199-1956, 2586-2430
243, 156

X = pymc.Uniform('x',70.,75.,value=74.4)
Y = pymc.Uniform('y',77.,82.,value=78.6)

GR = pymc.Uniform('re',0.5,150.,value=1.)
GQ = pymc.Uniform('q',0.2,1.,value=0.9)
GP = pymc.Uniform('pa',-180.,180.,value=110.)
GN = pymc.Uniform('gn',0.7,8.,value=3.)

LB = pymc.Uniform('b',5.,20.,value=10.2)
LQ = pymc.Uniform('q',0.2,1.,value=0.99)
LP = pymc.Uniform('pa',-180.,180.,value=110.)

SX = pymc.Uniform('sx',60.,190.,value=75.6+20.)
SY = pymc.Uniform('sy',60.,190.,value=82.6+13.)

SX = pymc.Uniform('sx',60.,190.,value=97.)
SY = pymc.Uniform('sy',60.,190.,value=95.)

SR = pymc.Uniform('sr',0.1,50.,value=1.)
SN = pymc.Uniform('sn',0.5,10.,value=3.)
SQ = pymc.Uniform('sq',0.2,1.,value=0.9)
SP = pymc.Uniform('sp',-180.,180.,value=10.)

LB2 = pymc.Uniform('lb2',1.,80.,value=25.)
LE = pymc.Uniform('le',0.4,1.6,value=1.)

XB = pymc.Uniform('xb',-0.2,0.2,value=0.)
XP = pymc.Uniform('xp',-180.,180.,value=-50.)

GR2 = pymc.Uniform('gr2',1.,1000.,value=25.)
GN2 = pymc.Uniform('gn2',0.7,7.,value=3.)


pars = [X,Y,GR,GQ,GP,GN,LB,LQ,LP,SX,SY,SR,SN,SQ,SP,LB2,LE]
pars += [XB,XP]



cov = [0.03,0.03,0.3,0.03,5.,0.1]
cov += [0.05,0.03,5.]
cov += [0.1,0.1,0.1,0.1]
cov += [0.03,10.]
cov += [0.3]
cov += [0.05]
cov += [0.005,7.]

gal = SBObjects.Sersic('',{'x':X,'y':Y,'re':GR,'q':GQ,'pa':GP,'n':GN})
gal2 = SBObjects.Sersic('',{'x':X,'y':Y,'re':GR2,'q':GQ,'pa':GP,'n':GN2})
src = SBObjects.Sersic('',{'x':SX,'y':SY,'re':SR,'q':SQ,'pa':SP,'n':SN})

lens = MassModels.PowerLaw('',{'x':X,'y':Y,'b':LB,'q':LQ,'pa':LP,'eta':LE})
lens2 = MassModels.PowerLaw('',{'x':75.1+243,'y':79.1+156,'q':1,'pa':0.,'b':LB2,'eta':1.})
shear = MassModels.ExtShear('',{'x':X,'y':Y,'b':XB,'pa':XP})
lenses = [lens,lens2,shear]

gals = [gal,gal2]
srcs = [src]

nit = 1
vals = [74.61003833445514, 79.839325373076, 27.78066646044798, 0.9618990419584786, 127.57456691957671, 5.750778290787367, 9.542125144567903, 0.9901110405962396, -47.42396298499143, 98.13830254140647, 97.29527621078647, 14.598756764471519, 6.317133455387623, 0.6884543411515842, 97.30523093899524, 26.369337562144437, 1.24977590350949, -0.036550619903778184, -1.4363330660255151]

vals = [74.61333001662433, 79.85927439175431, 26.035999528694333, 0.9796148468526131, 115.88114731345674, 5.80711750437209, 9.54768262871059, 0.9700921212375843, -31.436704131700846, 97.93722544778952, 97.20973235236619, 14.988026458948577, 6.217139315930574, 0.6766076824379276, 95.84149093719532, 26.152387219083888, 1.2457352050304835, -0.041016133694093765, -2.8816294451794073]
for i in range(len(vals)):
    pars[i].value = vals[i]#*(1.+numpy.random.randn()*0.01)

#Y.value = Y.value+0.3
GR.value = 6.
GN.value = 4.
GR2.value = 25.
GN2.value = 1.5
SR.value = 7.
SN.value = 5.
pars = [X,Y,GR,GQ,GP,GN,GR2,GN2,SR,SN]
cov = [0.03,0.03,0.1,0.03,5.,0.1,0.1,0.1,0.1,0.1]

vals = [74.6132630134146, 79.85900434768728, 2.318079124992558, 0.9861176991300905, 122.41153047750016, 1.295546674251775, 11.2627062126583, 0.726722739762727]
vals = [74.61327675532631, 79.85908511642121, 2.588857431174264, 0.9445339285209984, 118.93295331041483, 1.968767342290427, 22.309354144290204, 1.8772356356601296]
#vals = [74.61324829447115, 79.85903047075541, 6.309442264217598, 0.9758449218427194, 112.50620120664233, 3.2156859927573676, 14.230452048237645, 0.7236902049244635]
for i in range(len(vals)):
    pars[i].value = vals[i]


def getModel(getResid=False):
    for g in gals:
        g.setPars()
    for s in srcs:
        s.setPars()
    for l in lenses:
        l.setPars()

    lx,ly = pylens.getDeflections(lenses,[x,y])

    nmod = len(gals)+len(srcs)+1
    model = numpy.empty((sflt.size,nmod))
    for i in range(len(gals)):
        model[:,i] = convolve.convolve(gals[i].pixeval(x,y),psf,False)[0].ravel()/sflt
    for i in range(len(srcs)):
        model[:,i+len(gals)] = convolve.convolve(srcs[i].pixeval(lx,ly),psf,False)[0].ravel()/sflt
    model[:,-1] = 1./sflt

    fit,chi = optimize.nnls(model,rhs)
    if getResid is True:
        print fit,-0.5*chi**2
        return (model.T*sflt).T*fit
    return chi

@pymc.observed
def logl(value=0.,tmp=pars):
    return -0.5*getModel()**2





for i in range(nit):
    S = AMAOpt(pars,[logl],[],cov=numpy.array(cov))
    S.sample(1600)
    print S.logps[-1],S.trace[-1].tolist()
    S = AMAOpt(pars,[logl],[],cov=numpy.array(cov)/2.)
    S.sample(1600)
    print S.logps[-1],S.trace[-1].tolist()

    S = AMAOpt(pars,[logl],[],cov=numpy.array(cov)/4.)
    S.sample(1600)
    print S.logps[-1],S.trace[-1].tolist()
    S = AMAOpt(pars,[logl],[],cov=numpy.array(cov)/8.)
    S.sample(1600)
    print S.logps[-1],S.trace[-1].tolist()
#    S = AMAOpt(pars,[logl],[],cov=numpy.array(cov)/4.)
#    S.sample(2600)
#    print S.logps[-1],S.trace[-1].tolist()
    cov = numpy.array(cov)/2.


model = getModel(True).sum(1).reshape(img.shape)

#pylab.imshow(model,origin='lower',interpolation='nearest')
pylab.imshow((img-model)/sig,origin='lower',interpolation='nearest')
pylab.colorbar()
pylab.figure()
pylab.imshow(model,origin='lower',interpolation='nearest')
pylab.colorbar()

pylab.show()

