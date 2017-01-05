import numpy,pylab,time
import massmodel
import ndinterp
import indexTricks as iT
from scipy import interpolate
import special_functions as sf

q1 = 0.99999
r,e = iT.coords((3,180))
r = 10**(r-1)
e /= 181/(numpy.pi)
x = r*numpy.cos(e)
y = q1*r*numpy.sin(e)
g1 = 1.0
g2 = 1.0

var = {}
const = {'x':0.,'y':0.,'q':q1,'pa':0.,'b':1.,'eta':g1}
x = numpy.logspace(-3,1,41)
y = x*0.
PL = massmodel.PowerLaw('t',var,const)
PL.eta = 0.3
x1 = PL.deflections(x,y)[0]
PL.eta = 1.7
x2 = PL.deflections(x,y)[0]
print x1
print x2
print x1*x2

gs = numpy.linspace(0.4,1.,18)
g0 = numpy.empty(0)
r0 = g0*0.
d0 = g0*0.
for g in gs:
    PL.eta = g
    x1,y1 = PL.deflections(x,y)
    #PL.eta = 2.-g
    #x2,y2 = PL.deflections(x,y)
    #pylab.plot(x,x1*x2)
    g0 = numpy.concatenate((g0,x*0+g))
    r0 = numpy.concatenate((r0,x))
    d0 = numpy.concatenate((d0,x1))
pylab.scatter(numpy.log10(r0),g0,c=d0)
pylab.colorbar()
pylab.figure()
pylab.scatter(numpy.log10(r0),g0,c=r0**(1-g0))
pylab.colorbar()
pylab.show()
df

lo = []
hi = []
qs = numpy.linspace(0.2,0.99,80)
gs = numpy.linspace(0.2,1.9,180)
g = []
q = []
l = []
for g2 in gs:
    lo = []
    hi = []
    for q2 in qs:
        PL.eta = g2
        PL.q = q2
        x = r*numpy.cos(e)
        y = q2*r*numpy.sin(e)
        x2,y2 = PL.deflections(x,y)
        j = (x1/x2)[1]
        lo.append(j.min())
        hi.append((j.max()-j.min())/2.)
        g.append(g2)
        q.append(q2)
        #l.append(j.min())
        l.append(hi[-1])
    #pylab.plot(j)
#pylab.scatter(lo,hi,c=numpy.linspace(0.5,0.95,10))
    #pylab.plot(qs,hi)
#pylab.plot(qs,hi)
g = numpy.array(g)
q = numpy.array(q)
l = numpy.array(l)
fitdata = numpy.array([g,q,l]).T
print fitdata.shape
fitdata = fitdata[fitdata[:,2]>0.01]
print fitdata.shape
fit = sf.lsqfit(fitdata,'chebyshev',6,6)
guess = sf.genfunc(g,q,fit)
pylab.scatter(g,q,c=l)
pylab.colorbar()
pylab.figure()
c = l>0.05
pylab.scatter(g[c],q[c],c=(l-guess)[c]/l[c])
pylab.colorbar()
pylab.show()

q2 = 0.6
const = {'x':0.,'y':0.,'q':q2,'pa':0.,'b':1.,'eta':g2}
PL = massmodel.PowerLaw('t',var,const)
x = r*numpy.cos(e)
y = q2*r*numpy.sin(e)
x2,y2 = PL.deflections(x,y)
j = (x1/x2)[1]
print j.min(),j.max()
pylab.plot(j)

pylab.show()
df

y,x = iT.coords((162,162))
y /= 40.
x /= 40.
y -= 1.5*y[1,1]
x -= 1.5*x[1,1]

r = (x**2+y**2/0.723)**0.5
axes = {}
#axes[0] = interpolate.splrep(r[

ox = numpy.empty((162,162,17,33))
oy = numpy.empty((162,162,17,33))
q = numpy.linspace(0.2,1.,17)
e = numpy.linspace(0.2,1.8,33)
q[-1] = 0.999
var = {}
for i in range(q.size):
    for j in range(e.size):
        const = {'x':0.,'y':0.,'q':q[i],'pa':0.,'b':1.,'eta':e[j]}
        PL = massmodel.PowerLaw('t',var,const)
        x1,y1 = PL.deflections(x,y)
        ox[:,:,i,j] = x1
        oy[:,:,i,j] = y1
axes = {}
axes[0] = interpolate.splrep(x[0],numpy.arange(162),k=1)
axes[1] = interpolate.splrep(x[0],numpy.arange(162),k=1)
axes[2] = interpolate.splrep(q,numpy.arange(17),k=1)
axes[3] = interpolate.splrep(e,numpy.arange(33),k=1)

xmodel = ndinterp.ndInterp(axes,ox,order=1)
ymodel = ndinterp.ndInterp(axes,oy,order=1)

t = time.time()
for i in range(10):
    const = {'x':0.,'y':0.,'q':0.723,'pa':0.,'b':1.,'eta':1.076}
    PL = massmodel.PowerLaw('t',var,const)
    x1,y1 = PL.deflections(x,y)
print time.time()-t

t = time.time()
for i in range(10):
    points = numpy.array([x.ravel(),y.ravel(),x.ravel()*0.+0.723,y.ravel()*0.+1.076]).T
print time.time()-t
t = time.time()
for i in range(10):
    x0 = xmodel.eval(points).reshape(x.shape)
    y0 = ymodel.eval(points).reshape(y.shape)
print time.time()-t
print x0.min(),x0.max()
print x0,y0
res = (x0-x1)/x1
pylab.imshow(res[10:,10:])
pylab.colorbar()
pylab.show()





