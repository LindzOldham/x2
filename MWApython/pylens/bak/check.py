import numpy,sys,cPickle

n = 90000
x = numpy.random.random(n)*3.999+0.001
y = numpy.random.random(n)*3.999+0.001
q = numpy.random.random(n)*0.8+0.2
eta = numpy.random.random(n)*1.7 + 0.3

xgrid,ygrid = numpy.load('GRIDS')
xgrid.set_order(3)
a = numpy.empty((n,4))
a[:,0] = x.copy()
a[:,1] = y.copy()
a[:,2] = eta.copy()
a[:,3] = q.copy()
import time
t = time.time()
x0 = xgrid.eval(a)
y0 = ygrid.eval(a)
print time.time()-t

def getAngles(x,y,q,eta):
    from scipy import interpolate
    s = 1e-15
    if q==1:
        q = 0.9999999
    q2 = q**2
    b = 1-q2
    g = 0.5*eta-1.
    qb = ((2*x*y)/b)**g  # q_bar
    qt = qb*q*(x*y)**0.5/b  # q_tilde
    sb = 0.5*(x/y - y/x) + s**2*b/(2*x*y)
    nu1 = s**2*b/(2*x*y)
    nu2 = nu1+ 0.5*b*(x/y + y/(x*q2))
    nu = numpy.logspace(numpy.log10(nu1),numpy.log10(nu2),1001)
    mu = nu-sb
    t = (1+mu**2)**0.5
    f1 = (t-mu)**0.5/t
    f2 = (t+mu)**0.5/t
    ng = nu**g
    I1 = interpolate.splrep(nu,f1*ng)
    I2 = interpolate.splrep(nu,f2*ng)
    alphax = qt*interpolate.splint(nu1,nu2,I1)
    alphay = qt*interpolate.splint(nu1,nu2,I2)
    return alphax,alphay

dx = x*0.
dy = y*0.
for i in range(x.size):
    a,b = getAngles(x[i],y[i],q[i],eta[i])
    dx[i] = a
    dy[i] = b

o = (dx-x0)/dx
print eta[o<-2*o.std()]
print o.mean(),o.std()
print (dx-x0).mean(),(dx-x0).std()
import pylab
pylab.plot((dx-x0)/dx)
pylab.show()
