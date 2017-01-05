import untitled,numpy

c = [0.,0.]

q = 0.5

x = numpy.random.random(90000)*2 - 1.
y = numpy.random.random(90000)*2 - 1.
import time
t = time.time()
x0,y0 = untitled.powerlawdeflections(x,y,1.3,1.7,0.00001,q)
print time.time()-t

from pylens import *

var = {}
#const = {'x':0.,'y':0.,'q':0.5,'pa':0.,'b':1.3}
const = {'x':0.,'y':0.,'q':q,'pa':90.,'b':1.3,'eta':1.7}
PL = massmodel.PowerLaw('t',var,const)
t = time.time()
x1,y1 = PL.deflections(x,y)
print time.time()-t
print x0/x1
