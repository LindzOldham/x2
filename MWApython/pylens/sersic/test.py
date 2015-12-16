import pylab,numpy
import indexTricks as iT
from scipy import ndimage
import ndinterp
from scipy.special import gamma,gammainc
import sersic
from pylens import powerlaw

x0 = numpy.logspace(-4.,1.,51)
#x0 = numpy.linspace(0.1,15.,151)
x,y = ndinterp.create_axes_array([x0,x0])

N = 3.7
Q = 0.73

k = 2.*N-1./3+4./(405.*N)+46/(25515.*N**2)
amp = k**(2*N)/(2*N*gamma(2*N))
print amp,x.size
amp = 1.
import time
t = time.time()
yi,xi = sersic.sersicdeflections(-y.ravel(),x.ravel(),amp,1.,N,Q)
yi *= -1
print time.time()-t
t = time.time()
yy,xx = powerlaw.powerlawdeflections(x.ravel(),y.ravel(),2.4,1.1,1e-5,0.73)
print time.time()-t
xr = x.ravel()
yr = y.ravel()
t = time.time()
gam = 1.1/2.
s2 = 1e-5*1e-5
q0 = (1.-gam)*2.4**(2*gam)/(0.73**gam)
defs = [powerlaw.fastelldefl(xr[i],yr[i],q0,gam,0.73,s2) for i in range(x.size)]
print time.time()-t
t = time.time()
phi = sersic.sersicpotential(-y.ravel(),x.ravel(),amp,1.,N,Q)
dy,dx = numpy.gradient(phi.reshape(x.shape))
dy /= x
dx /= y
dx /= 0.1
dx /= numpy.log(10)
print time.time()-t

yi = yi.reshape(x.shape)
xi = xi.reshape(x.shape)

#pylab.imshow(phi.reshape(x.shape),origin='lower')
pylab.imshow((dx/yi-1.)[1:-1,1:-1],origin='lower')
pylab.colorbar()
pylab.figure()
pylab.imshow(dy,origin='lower')
pylab.colorbar()
pylab.figure()
pylab.imshow(yi.reshape(x.shape),origin='lower')
pylab.colorbar()
pylab.figure()
pylab.imshow(xi.reshape(x.shape),origin='lower')
pylab.colorbar()
#pylab.imshow((xi**2+yi**2).reshape(x.shape)**0.5,origin='lower')
pylab.show()
