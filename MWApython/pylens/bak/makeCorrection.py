import numpy,pylab,time,cPickle
import massmodel
import ndinterp
import indexTricks as iT
from scipy import interpolate,special
import special_functions as sf


q1 = 0.99999
r,e = iT.coords((3,180))
if r.shape[0]==3:
    r = 10**(r-1)
else:
    r = 10**(r/10.-3)
e /= 181/(numpy.pi)
x = r*numpy.cos(e)
y = r*numpy.sin(e)*q1
g1 = 1.0
g2 = 1.0

var = {}
const = {'x':0.,'y':0.,'q':q1,'pa':0.,'b':1.,'eta':g1}
PL = massmodel.PowerLaw('t',var,const)
x1,y1 = PL.deflections(x,y)

qs = numpy.linspace(0.2,0.99,80)
gs = numpy.linspace(0.2,1.9,180)
g = []
q = []
amp = []
offset = []
oldratio = None
form = numpy.load('blah')
for g2 in gs:
    corrs = {}
    for q2 in qs:
        c = r**(g2-1)
        PL.eta = g2
        PL.q = q2
        x = r*numpy.cos(e)#/q2**0.5
        y = r*numpy.sin(e)*q2
        x2,y2 = PL.deflections(x,y)
        x2 *= c
        j = (x1/x2)[1]
        offset.append(j.min())
        amp.append((j.max()-j.min())/2.)
        #pylab.plot(j)
        model = amp[-1]*form[q2]+offset[-1]
        model = amp[-1]*numpy.cos(e[1]*2)+offset[-1]+amp[-1]
        fitdata = numpy.array([e[1],j-model]).T
        fit = sf.lsqfit(fitdata,'chebyshev',5)
        model2 = sf.genfunc(e[1],0,fit)
        #pylab.plot(j/(model+model2))
        print amp[-1],offset[-1]
        #model += numpy.cos(e[1]*4)*-0.064+0.064
        #model += numpy.sin(e[1]*4)*-0.03
        model += numpy.cos(e[1]*4)*-amp[-1]/4.+amp[-1]/4.
        model += numpy.sin(e[1]*4)*-amp[-1]/8
        model += numpy.sin(e[1]*8)*amp[-1]/16
        model = (special.j0(e[1]*2.355)+0.407)*amp[-1]*1.5#+offset[-1]
        pylab.plot(j-offset[-1])#-model)#+numpy.sin(e[1]*2)*0.032-0.032)
        pylab.plot(model)
        #pylab.plot(numpy.sin(e[1]*8)*amp[-1]/16)
        #pylab.plot(model)
        #pylab.plot(model+model2)
        pylab.show()
        #pylab.plot(model)
        #pylab.show()
        #pylab.plot(j/model)
        #pylab.plot(1.0063-0.0063*numpy.cos(e[1]*4))
        #pylab.plot(model)
        #pylab.plot(amp[-1]*numpy.cos(e[1]*2)+offset[-1]+amp[-1]-j)
        #pylab.show()
        g.append(g2)
        q.append(q2)
    pylab.show()
ampfit,offsetfit,g,q,amp,offset = numpy.load('FITS')

g = numpy.array(g)
q = numpy.array(q)
amp = numpy.array(amp)
fitdata = numpy.array([g,q,amp]).T
ampfit = sf.lsqfit(fitdata,'chebyshev',4,4)
guess = sf.genfunc(g,q,ampfit)
pylab.scatter(g,q,c=amp)
pylab.colorbar()
pylab.figure()
pylab.scatter(g,q,c=(amp-guess)/amp)
pylab.colorbar()

offset = numpy.array(offset)
fitdata = numpy.array([g,q,offset]).T
offsetfit = sf.lsqfit(fitdata,'chebyshev',4,4)
guess = sf.genfunc(g,q,offsetfit)
pylab.figure()
pylab.scatter(g,q,c=offset)
pylab.colorbar()
pylab.figure()
pylab.scatter(g,q,c=(offset-guess)/offset)
pylab.colorbar()
pylab.show()

f = open('FITS','wb')
cPickle.dump([ampfit,offsetfit,g,q,amp,offset],f,2)
f.close()


