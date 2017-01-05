from pylens import MassModels
import numpy,pylab
import indexTricks as iT

xc = 52.3
yc = 47.8
b = 27.4
q = 0.9
b0 = b/q**0.5

d = MassModels.PowerLaw('a',{'x':xc,'y':yc,'b':b,'q':q,'pa':0.,'eta':1.})

y,x = iT.coords((100,100))

dx,dy = d.deflections(x,y)

from scipy import interpolate
u = numpy.logspace(-5,0,501)
y0,x0 = iT.coords((40,40))
y0 /= 10.
x0 /= 10.

xgrid = x0*0.
ygrid = x0*0.
d = (1-(1-q**2)*u)
for i in range(40):
    for j in range(40):
        xi2 = u*(x0[0,i]**2+y0[j,0]**2/d)
        k = 1./xi2
        k /= 0.5*b0
        I = k/d**0.5
        mod = interpolate.splrep(u,I)
        xgrid[i,j] = q*x0[0,i]*interpolate.splint(0,1,mod)

pylab.imshow(xgrid,origin='lower')
pylab.colorbar()
pylab.show()
xgrid[numpy.isnan(xgrid)] = 0.
X = (x-xc)/b0
Y = (y-yc)/b0

axes = {}
axes[0] = interpolate.splrep(x0[0],numpy.arange(40),k=1)
axes[1] = interpolate.splrep(x0[0],numpy.arange(40),k=1)

import ndinterp
xmod = ndinterp.ndInterp(axes,xgrid)
mx = (xmod.eval(numpy.array([abs(X).flatten(),abs(Y).flatten()]).T)*b0).reshape(X.shape)*numpy.sign(X)
print mx
print dx
print mx/dx

import pylab
pylab.imshow(mx/dx,origin='lower')
pylab.colorbar()
pylab.show()

q = numpy.linspace(0.1,1.,19)
n = numpy.linspace(0.4,6.,29)
grid = numpy.empty((40,40,q.size,n.size))


