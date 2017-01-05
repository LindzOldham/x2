import numpy,pylab,time,cPickle
import massmodel
import ndinterp
import indexTricks as iT
from scipy import interpolate
import special_functions as sf

q1 = 0.99999
r,e = iT.coords((41,180))
r = 10**(r/40-3)
e /= 181/(numpy.pi)
x = r*numpy.cos(e)
y = q1*r*numpy.sin(e)
g1 = 1.0
g2 = 1.0

var = {}
const = {'x':0.,'y':0.,'q':q1,'pa':0.,'b':1.,'eta':g1}
PL = massmodel.PowerLaw('t',var,const)
x1,y1 = PL.deflections(x,y)

"""
lo = []
hi = []
qs = numpy.linspace(0.2,0.99,80)
gs = numpy.linspace(0.2,1.9,180)
g = []
q = []
amp = []
offset = []
for g2 in gs:
    for q2 in qs:
        PL.eta = g2
        PL.q = q2
        x = r*numpy.cos(e)
        y = q2*r*numpy.sin(e)
        x2,y2 = PL.deflections(x,y)
        j = (x1/x2)[1]
        offset.append(j.min())
        amp.append((j.max()-j.min())/2.)
        g.append(g2)
        q.append(q2)

g = numpy.array(g)
q = numpy.array(q)
amp = numpy.array(amp)
fitdata = numpy.array([g,q,amp]).T
ampfit = sf.lsqfit(fitdata,'chebyshev',6,6)
offset = numpy.array(offset)
fitdata = numpy.array([g,q,offset]).T
offsetfit = sf.lsqfit(fitdata,'chebyshev',6,6)

f = open('FITS','wb')
cPickle.dump([ampfit,offsetfit,g,q,amp,offset],f,2)
f.close()
"""
ampfit,offsetfit = numpy.load('FITS')[:2]

qt = 0.743
gt = 1.058
#bt = 30.4
#PL.b = bt
PL.eta = gt
PL.q = qt

x = r*numpy.cos(e)
y = q1*r*numpy.sin(e)

amp = sf.genfunc(gt,qt,ampfit)
offset = sf.genfunc(gt,qt,offsetfit)
corr = numpy.cos(e)*amp+offset
d_a = r**(1.-gt)/corr
D = numpy.cos(numpy.arctan(y/x))*d_a
D[x<0] *= -1
x1,y1 = PL.deflections(x,y)

pylab.imshow((D-x1)/x1)
pylab.colorbar()
pylab.figure()
pylab.imshow(x1)
pylab.colorbar()
pylab.show()


y,x = iT.coords((320,320))
x -= x.mean()
y -= y.mean()

qt = 0.743
gt = 1.058
bt = 30.4
e = numpy.arctan(y/x/qt)
e[x<0] += numpy.pi
r2 = (x**2+y**2/qt**2)
x0 = numpy.cos(e)*r2**0.5
y0 = qt*numpy.sin(e)*r2**0.5
pylab.imshow(x-x0)
pylab.colorbar()
pylab.figure()
pylab.imshow(y-y0)
pylab.colorbar()
pylab.show()

d_a = bt*bt**(gt-1)/r2**((gt-1)/2.)
amp = sf.genfunc(gt,qt,ampfit)
offset = sf.genfunc(gt,qt,offsetfit)
corr = numpy.cos(e)*amp+offset
D = numpy.cos(numpy.arctan(y/x))*d_a/corr
D[x<0] *= -1

x1,y1 = PL.deflections(x,y)

pylab.figure()
pylab.imshow(D)
pylab.colorbar()
pylab.figure()
pylab.imshow(x1)
pylab.colorbar()
pylab.figure()
pylab.imshow((D-x1)/x1)
pylab.colorbar()
pylab.show()


