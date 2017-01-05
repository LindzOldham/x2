import numpy,pylab,time,cPickle
import massmodel
import ndinterp
import indexTricks as iT
from scipy import interpolate
import special_functions as sf


q1 = 0.99999
r,e = iT.coords((41,180))
#r = 10**(r-1)
r = 10**(r/10.-3)
e /= 181/(numpy.pi)
x = r*numpy.cos(e)
y = q1*r*numpy.sin(e)
g1 = 1.0
g2 = 1.0

var = {}
const = {'x':0.,'y':0.,'q':q1,'pa':0.,'b':1.,'eta':g1}
PL = massmodel.PowerLaw('t',var,const)
x1,y1 = PL.deflections(x,y)

ampfit,offsetfit = numpy.load('FITS')[:2]

qt = 0.743
gt = 1.058
#bt = 30.4
#PL.b = bt
PL.eta = gt
PL.q = qt

x = r*numpy.cos(e)
y = qt*r*numpy.sin(e)

x2,y2 = PL.deflections(x,y)

amp = sf.genfunc(gt,qt,ampfit)
offset = sf.genfunc(gt,qt,offsetfit)
corr = numpy.cos(e)*amp+offset
d_a = r**(1.-gt)*corr
D = numpy.cos(numpy.arctan2(y,x))*d_a
#D[x<0] *= -1
pylab.imshow(x2/D)
pylab.colorbar()
#pylab.plot((x2/D)[7])
pylab.show()




ratio = x1/x2
ratio /= 10**(numpy.log10(r)*(gt-1))
#print numpy.log10(r[:,0]),numpy.log10(ratio[:,0])
pylab.plot(numpy.log10(r[:,0]),numpy.log10(ratio[:,0]))
pylab.show()
#for i in range(ratio.shape[1]):
#    pylab.plot(ratio[:,i])
#pylab.show()
pylab.imshow(ratio)
pylab.colorbar()
pylab.show()


