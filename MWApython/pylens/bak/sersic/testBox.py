from pylens import MassModels
import numpy,pylab
import indexTricks as iT
from scipy import interpolate
from scipy.special import gamma,gammainc
import ndinterp,time
import sersic
import appSersic
import qsersic

xc = 52.3
yc = 47.8
b = 45.6628
qin = 0.753
reff = 15.13
n = 2.5373

y,x = iT.coords((100,100))

t = time.time()
long = MassModels.Sersic('f',{'x':xc,'y':yc,'q':qin,'pa':0.,'b':b,'reff':reff,'n':n})
for i in range(10):
    x0,y0 = long.deflections(x,y)
print time.time()-t

t = time.time()
ybox,xbox = numpy.load('serModels.dat')
for i in range(10):
    X = (x-xc).flatten()
    Y = (y-yc).flatten()
    xs = numpy.sign(X)
    ys = numpy.sign(Y)
    reff0 = reff/qin**0.5
    k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
    amp = (b/reff)**2/gammainc(2*n,k*(b/reff)**(1/n))

    eval = numpy.array([abs(Y)/reff0,abs(X)/reff0,Y*0+n,Y*0+qin]).T
    x1 = (amp*xs*xbox.eval(eval)).reshape(x.shape)*reff0
    y1 = (amp*ys*ybox.eval(eval)).reshape(y.shape)*reff0
print time.time()-t

t = time.time()
short = MassModels.SersicG('f',{'x':xc,'y':yc,'q':qin,'pa':0.,'b':b,'reff':reff,'n':n})
for i in range(10):
    x2,y2 = short.deflections(x,y)
print time.time()-t
print x1/x2
print y1/y2
pylab.imshow(y0/y2,origin='lower')
pylab.colorbar()
pylab.show()
