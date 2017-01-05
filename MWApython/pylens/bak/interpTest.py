import numpy,pylab,time
import massmodel
import ndinterp
import indexTricks as iT
from scipy import interpolate

y,x = iT.coords((162,162))
y /= 40.
x /= 40.
y -= 1.5*y[1,1]
x -= 1.5*x[1,1]

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





