import numpy,pyfits,pylab
import pymc
from SampleOpt import AMAOpt
from scipy import optimize,ndimage


models = numpy.load('miles_EELs_o3.dat')
img = '/data/ljo31b/EELs/esi/kinematics/apertures/final/J0837_ap_1.00_spec_lens.fits'
var = '/data/ljo31b/EELs/esi/kinematics/apertures/final/J0837_ap_1.00_var_lens.fits'
wave = '/data/ljo31b/EELs/esi/kinematics/apertures/final/J0837_ap_1.00_wl_lens.fits'
img = pyfits.open(img)[0].data
var = pyfits.open(var)[0].data
wave = 10**pyfits.open(wave)[0].data
sig = var**0.5


good = numpy.isfinite(img)

img = img[good]
var = var[good]
wave = wave[good]

z1 = 0.42485
Z1 = pymc.Uniform('',z1-0.001,z1+0.001,value=z1)
S1 = pymc.Uniform('',125.,415,value=220.)


w1 = 3850.
w2 = 4800.
c = (wave/(1.+z1)>w1)&(wave/(1.+z1)<w2)

z2 = 0.64069
Z2 = pymc.Uniform('',z2-0.001,z2+0.001,value=z2)
S2 = pymc.Uniform('',125.,415,value=220.)


#w1 = 3850
w2 = 4800
c = (wave/(1.+z1)>w1)&(wave/(1.+z2)<w2)

D = img[c]
W = wave[c]
S = sig[c]

X = numpy.arange(D.size)
X = 2.*X/X.max()-1.

c = (W<6855)|(W>6925)
c = c&((W<7590)|(W>7730))
#c = c&((W<5800)|(W>5900))
#c = c&((W<6550)|(W>6620))

D = D[c]
W = W[c]
S = S[c]

norder = 7
linmodel = numpy.empty((norder+1+len(models)*2,D.size))
X = numpy.arange(D.size)
X = 2.*X/X.max()-1.
for i in range(norder+1):
    linmodel[i] = (X**i)/S

R = D/S
R = D.copy()
#for i in range(norder+1):
#    R += 5*X**(i)
R = R/S
@pymc.observed
def logl(value=0.,z=Z1,s=S1,zz=Z2,ss=S2):
    w = W/(1+z)
    pnts = numpy.array([w*0.+s,w]).T
    for i in range(len(models)):
        linmodel[norder+1+i] = models[i].eval(pnts)/S

    w = W/(1+zz)
    pnts = numpy.array([w*0.+ss,w]).T
    for i in range(len(models)):
        linmodel[norder+1+i+len(models)] = models[i].eval(pnts)/S

    lhs = numpy.dot(linmodel,linmodel.T)
    rhs = numpy.dot(linmodel,R)

    fit = numpy.linalg.solve(lhs,rhs)
    res = -0.5*((numpy.dot(linmodel.T,fit)-R)**2).sum()
    return res


SS = AMAOpt([Z1,S1,Z2,S2],[logl],cov=[0.0001,10.,0.0001,10.])
SS.sample(500)
print SS.logps[-1],SS.trace[-1]
SS.sample(500)
print SS.logps[-1],SS.trace[-1]


z,s,zz,ss = SS.trace[-1]

if 1==1:
    w = W/(1+z)
    pnts = numpy.array([w*0.+s,w]).T
    for i in range(len(models)):
        linmodel[norder+1+i] = models[i].eval(pnts)/S

    w = W/(1+zz)
    pnts = numpy.array([w*0.+ss,w]).T
    for i in range(len(models)):
        linmodel[norder+1+i+len(models)] = models[i].eval(pnts)/S

    lhs = numpy.dot(linmodel,linmodel.T)
    rhs = numpy.dot(linmodel,R)

    fit = numpy.linalg.solve(lhs,rhs)
    res = -0.5*((numpy.dot(linmodel.T,fit)-R)**2).sum()

    print fit
    linmodel *= S
    R = numpy.dot(linmodel.T,fit)
    pylab.plot(W,D)
    pylab.plot(W,numpy.dot(linmodel.T,fit))
    pylab.plot(W,D-R)
    pylab.plot(W,ndimage.uniform_filter(D-R,5))
    pylab.show()
    df


    w = W/(1+z)
    pnts = numpy.array([w*0.+s,w]).T
    for i in range(len(models)):
        linmodel[norder+1+i] = models[i].eval(pnts)/S

    fit,chi = optimize.nnls(linmodel.T,R)
    print fit
    linmodel *= S
    for i in range(norder+1):
        fit[i] = fit[i]-5
    R = numpy.dot(linmodel.T,fit)
    pylab.plot(W,numpy.dot(linmodel.T,fit))
    pylab.plot(W,D)
    pylab.plot(W,D-R)
    pylab.show()
df

R = D/S
if 1==1:
    w = W/(1+z)
    pnts = numpy.array([w*0.+s,w]).T
    for i in range(len(models)):
        linmodel[norder+1+i] = models[i].eval(pnts)/S

    lhs = numpy.dot(linmodel,linmodel.T)
    rhs = numpy.dot(linmodel,R)

    fit = numpy.linalg.solve(lhs,rhs)
    res = -0.5*((numpy.dot(linmodel.T,fit)-R)**2).sum()
    print fit
    linmodel *= S
    pylab.plot(w,numpy.dot(linmodel.T,fit))
    pylab.plot(w,D)
    pylab.show()

