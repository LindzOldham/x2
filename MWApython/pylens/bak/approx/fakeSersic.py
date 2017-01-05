from pylens import MassModels
import numpy,pylab,time
import indexTricks as iT
from scipy.special import gamma,gammainc
from imageSim import SBModels
from scipy import ndimage

xc = 52.3
yc = 47.8
b = 45.6628
qin = 0.753
reff = 15.13
n = 2.5373

ntime = 10

y,x = iT.coords((100,100))

t = time.time()
long = MassModels.Sersic('f',{'x':xc,'y':yc,'q':qin,'pa':0.,'b':b,'reff':reff,'n':n})
for i in range(ntime):
    x0,y0 = long.deflections(x,y)
print time.time()-t

t = time.time()
ybox,xbox = numpy.load('../serModelsHDR.dat')
for i in range(ntime):
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
X = iT.resamp(x,4).flatten()
Y = iT.resamp(y,4).flatten()

dx = numpy.empty((x0.size,X.size))
dy = dx*0.
xx = x.flatten()
yy = y.flatten()
for i in range(X.size):
    dx[:,i] = xx-X[i]
    dy[:,i] = yy-Y[i]
dr2 = (dx**2+dy**2)
dr = dr2**0.5
AX = dx/dr
AY = dy/dr
AX *= (1.-numpy.exp(-0.5*dr2/16.))/dr
AY *= (1.-numpy.exp(-0.5*dr2/16.))/dr
mass = SBModels.Sersic('t',{'x':xc,'y':yc,'q':qin,'pa':0.,'re':reff,'n':n})
for i in range(ntime):
    m = mass.pixeval(X,Y)*10.
    ax = (m*AX).sum(1).reshape(x.shape)
    ay = (m*AY).sum(1).reshape(y.shape)
print time.time()-t

pylab.imshow(x1,origin='lower',interpolation='nearest')
pylab.colorbar()
pylab.figure()
pylab.imshow(ax.reshape(x.shape),origin='lower',interpolation='nearest')
pylab.colorbar()
pylab.figure()
pylab.imshow(x1/ax,origin='lower',interpolation='nearest',vmin=0.8,vmax=1.2)
pylab.colorbar()

pylab.show()
