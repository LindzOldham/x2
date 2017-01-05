import powerlaw,numpy
import massmodel,pylab

c = [0.,0.]

q = 0.5

x = numpy.random.random(90000)*2 - 1.
y = numpy.random.random(90000)*2 - 1.
import indexTricks as iT
y,x = iT.overSample((320,320),1)
print y.shape
y -= y.mean()
x -= x.mean()
y /= 40.
x /= 40.
print x.min(),x.max()
import time
t = time.time()
x0,y0 = powerlaw.powerlawdeflections(x.ravel(),y.ravel(),1.3,1.,0.00001,q)
print time.time()-t

from pylens import *
var = {}
t = time.time()
const = {'x':0.,'y':0.,'q':q,'pa':0.,'b':1.3,'eta':0.9}
PL0 = massmodel.PowerLaw('t',var,const)
x0,y0 = PL0.deflections(x,y)

pylab.imshow(x0)
pylab.colorbar()
pylab.show()

print time.time()-t

var = {}
#const = {'x':0.,'y':0.,'q':0.5,'pa':0.,'b':1.3}
const = {'x':0.,'y':0.,'q':q,'pa':0.,'b':1.3,'eta':1.}
PL = massmodel.PowerLaw('t',var,const)
t = time.time()
x1,y1 = PL.deflections(x,y)
print time.time()-t
print x0/x1
xa,ya = PL.align_coords(x,y)
r1 = (q*xa**2+ya**2/q)**0.5
xa,ya = PL.align_coords(x0,y0)
a = (x0**2+y0**2)**0.5
import pylab
pylab.plot(r1,a,'ko')
pylab.show()
