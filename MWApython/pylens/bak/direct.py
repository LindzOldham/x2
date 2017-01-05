import numpy,pylab
from scipy import interpolate,optimize

mu = numpy.logspace(-3,0,601)
#mu = numpy.logspace(0,3,601)
mu = numpy.logspace(-3,3,601)
fmu = numpy.sqrt((1+mu**2)**-0.5 - mu/(mu**2+1))
mfmu = numpy.sqrt((1+mu**2)**-0.5 + mu/(mu**2+1))
#pylab.plot(mu,fmu)
#pylab.plot(mu,mfmu)

def fit(p,a,x,y):
    b,c,n = p
    #n = 0.4551*2**(b)
    f = n/x**a/(1+x**c)**b
    return numpy.log10(f)-y

q = 0.7
i = 1.
x2 = 0.4
for x1 in numpy.logspace(-1,1,3):
    #for x2 in numpy.logspace(-1,1,3):
    for q in [0.2,0.5,0.8]:
        r2 = x1**2+x2**2
        rho = numpy.logspace(-5,numpy.log10(r2**0.5),401)
        rhosinB = rho**2*(1-q**2)
        D2 = (rhosinB+x2**2-x1**2)**2+4*x1**2*x2**2
        D = D2**0.5
        w2 = (D+r2+rhosinB)/(D+r2-rhosinB)
        w = w2**0.5
        I1 = rho*w/(x1**2+w2**2*x2**2)
        #pylab.figure()
        #for i in numpy.linspace(0.5,1.5,11):
        I = x1*I1/rho**i
        pylab.plot(rho,I)
        #pylab.title('%f  %f'%(x1,x2))
pylab.show()


s = 0.005
nu = mu
#for i in numpy.linspace(0.5,1.5,11):
i = 1.
o = []
for s in numpy.logspace(-5,1,601):
    snu = nu-s
    fsnu = numpy.sqrt((1+snu**2)**-0.5 - snu/(snu**2+1))
    M = fsnu*nu**-i
    lM = numpy.log10(M)
    #pylab.plot(numpy.log10(mu),lM)
    coeff,ier = optimize.leastsq(fit,[i+1,1.,0.4551],(i,mu,lM))
    o.append(coeff)
    continue
    print i,coeff
    m = fit(coeff,i,mu,lM)
    print (m**2).sum()
    m = 10**(m+numpy.log10(M))
    pylab.plot(numpy.log10(mu),numpy.log10(m))
    #M = 0.9792/mu**i/(1+mu**1.327)**1.18
    #pylab.plot(numpy.log10(mu),numpy.log10(M))
    pylab.show()
o = numpy.array(o)
print o.shape
s = numpy.logspace(-5,1,601)
pylab.plot(s,o[:,0])
pylab.show()
df

s = 0.05

for i in numpy.linspace(0.5,1.5,11):
    smu = mu+s
    fmu = numpy.sqrt((1+mu**2)**-0.5 - mu/(mu**2+1))
    M = fmu*smu**-i
    lM = numpy.log10(M)
    pylab.plot(numpy.log10(mu),lM)
    coeff,ier = optimize.leastsq(fit,[i+1,1.,0.4551],(i,s,mu,lM))
    print i,coeff
    m = fit(coeff,i,s,mu,lM)
    print (m**2).sum()
    m = 10**(m+numpy.log10(M))
    pylab.plot(numpy.log10(mu),numpy.log10(m))
    #M = 0.9792/mu**i/(1+mu**1.327)**1.18
    #pylab.plot(numpy.log10(mu),numpy.log10(M))
    pylab.show()
pylab.show()


"""
lmu = numpy.log10(mu)
for i in numpy.linspace(0.5,1.5,11):
    smu = s-mu
    fmu = numpy.sqrt((1+smu**2)**-0.5 - smu/(smu**2+1))
    M = fmu*mu**-i
    lM = numpy.log10(M)
    pylab.plot(lmu,lM)
    #continue
    coeff,ier = optimize.leastsq(fit,[i+0.5,1.,1.09],(i,s,mu,lM))
    print i,coeff
    #coeff = [1./3,1.5,1.414]
    #coeff = [-0.2354,0.5127,0.9866]
    lm = fit(coeff,i,s,mu,lM)+lM
    pylab.plot(lmu,lm)
    pylab.show()
"""

import indexTricks as iT
import time
from scipy.special import hyp2f1
y,x = iT.coords((150,150))
#y -= y.mean()
#x -= x.mean()
y += 1.
x += 1
q = 0.7
t = time.time()
mu1 = 0.5*(y/x-x/y)
mu2 = 0.5*(y/(x*q**2) - x*q**2/y)
pylab.imshow(mu2-mu1)
print (mu2/mu1).min()
pylab.colorbar()
pylab.show()
f1b = -1.18
f1c = 1.327
def f1(x,g):
    return hyp2f1(f1b,(1-g)/f1c,1+(1-g)/f1c,-x**f1c)*x**(1-g)/g-1

dx = (f1(mu1,1.1)-f1(mu2,1.1))*q*(x*y)**0.5/(1-q**2)
print dx,mu1.max(),mu1.min(),mu2.max(),mu2.min()
