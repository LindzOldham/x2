from pylens import MassModels
import numpy,pylab,time
import indexTricks as iT
from scipy.special import gamma,gammainc
from imageSim import SBModels,convolve

N = 2
xc = 52.1*N
yc = 47.2*N
b = 45.6628
qin = 0.753
reff = 15.13
n = 2.5373

ntime = 1

y,x = iT.coords((100*N,100*N))

M = MassModels.PowerLaw('bah',{'x':xc,'y':yc,'b':b,'q':qin,'pa':0.,'eta':1.})

psf = numpy.random.random(961).reshape((31,31))
S = SBModels.Sersic('bah',{'x':xc,'y':yc,'re':1.,'pa':0.,'q':qin,'n':2.})

t = time.time()
for i in range(10):
    x0,y0 = M.deflections(x,y)
print time.time()-t

M.eta = 1.01
t = time.time()
for i in range(10):
    x0,y0 = M.deflections(x,y)
print time.time()-t

P = convolve.convolve(x,psf)[1]
t = time.time()
for i in range(10):
    J = convolve.convolve(S.pixeval(x,y,csub=1),P,False)[0]
print time.time()-t

