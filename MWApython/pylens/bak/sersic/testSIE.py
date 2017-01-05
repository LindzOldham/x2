from pylens import MassModels
import numpy,pylab
import indexTricks as iT
from scipy import interpolate
import ndinterp

xc = 52.3
yc = 47.8
b = 27.4
q = 0.997
b0 = b/q**0.5

eta = 0.5
d = MassModels.PowerLaw('a',{'x':xc,'y':yc,'b':b,'q':q,'pa':0.,'eta':eta})

y,x = iT.coords((100,100))

dx,dy = d.deflections(x,y)

from scipy import interpolate
u = numpy.logspace(-10,0,501)

x0 = numpy.logspace(-3,1,41)
y0 = numpy.logspace(-3,1,41)

x0,y0 = ndinterp.create_axes_array([x0,y0])

xgrid = x0*0.
ygrid = x0*0.
d = (1-(1-q**2)*u)
for i in range(41):
    for j in range(41):
        xi2 = u*(x0[i,0]**2+y0[0,j]**2/d)
        k = 1./xi2**eta
        k *= 0.5/(2-eta)
        I = k/d**0.5
        mod = interpolate.splrep(u,I)
        #pylab.plot(u,I)
        #print interpolate.splint(0.,1.,mod),interpolate.splint(0.001,1.,mod)
        #pylab.show()
        xgrid[i,j] = q*x0[i,0]*interpolate.splint(0.,1.,mod)

pylab.imshow(xgrid,origin='lower')
pylab.colorbar()
pylab.show()
xgrid[numpy.isnan(xgrid)] = 0.
X = (x-xc)/b0
Y = (y-yc)/b0

print abs(X).min()
print abs(Y).min()
axes = {}
axes[0] = interpolate.splrep(y0[0],numpy.arange(41),k=3)
axes[1] = interpolate.splrep(y0[0],numpy.arange(41),k=3)

import ndinterp
xmod = ndinterp.ndInterp(axes,xgrid)
mx = (xmod.eval(numpy.array([abs(X).flatten(),abs(Y).flatten()]).T)*b0).reshape(X.shape)*numpy.sign(X)
print mx
print dx
print mx/dx

import pylab
pylab.imshow(numpy.log10(mx/dx),origin='lower')
pylab.colorbar()
pylab.show()

q = numpy.linspace(0.1,1.,19)
n = numpy.linspace(0.4,6.,29)
grid = numpy.empty((40,40,q.size,n.size))


