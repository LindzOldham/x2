import pymc
import numpy
from scipy.integrate import quad
from stellarpop import distances
from math import pi
import pyfits
from scipy import ndimage,signal
from radbspline import pyfit
from pylens import *
from stellarpop import MyStepMethods as MySM

pl = massmodel.PowerLaw(1.,load=False)
es = massmodel.ExtShear()

gal = profiles.Sersic()
src = profiles.Sersic()

OVRS = 2

image = pyfits.open('sci.fits')[0].data.copy()
var = pyfits.open('sig.fits')[0].data.copy()
psf = pyfits.open('modpsf.fits')[0].data.copy()
psf /= psf.max()
psf /= psf.sum()
psfFFT = None

var = var**2

coords = numpy.indices((image.shape[0]*OVRS,image.shape[1]*OVRS))
xc = coords[1].copy()
yc = coords[0].copy()
del coords

#
# Define priors
#

covar = numpy.zeros((31,31))
# The lens model
lens_x = pymc.Uniform('lens_x',59.0,74.0,value=66.526300)
covar[0,0] = 0.5
lens_y = pymc.Uniform('lens_y',49.0,69.0,value=58.893339)
covar[1,1] = 0.5
lens_b = pymc.Uniform('lens_b',20.,27.,value=23.023903)
covar[2,2] = 0.3
lens_q = pymc.Uniform('lens_q',0.2,1.,value=0.364555)
covar[3,3] = 0.3
lens_t = pymc.Uniform('lens_t',0.,2*pi,value=1.918038)
covar[4,4] = 0.5

# The external shear
shear = pymc.Uniform('shear',0.,0.5,value=0.004265)
covar[5,5] = 0.1
shear_pa = pymc.Uniform('shear_pa',0.,2*pi,value=1.338236)
covar[6,6] = 0.5

# The source model
src_x = pymc.Uniform('src_x',54.0,74.0,value=63.730004)
covar[7,7] = 0.5
src_y = pymc.Uniform('src_y',49.0,69.0,value=57.589373)
covar[8,8] = 0.5
src_q = pymc.Uniform('src_q',0.2,1.,value=1.)
covar[9,9] = 0.3
src_t = pymc.Uniform('src_t',0.,2*pi,value=0)
covar[10,10] = 0.5
src_re = pymc.Uniform('src_re',3.,50.,value=13.619490)
covar[11,11] = 1.
src_amp = pymc.Uniform('src_amp',0.,0.2,value=0.021862)
covar[12,12] = 0.0001
src_n = pymc.Uniform('src_n',0.8,6.5,value=1.413759)
covar[13,13] = 0.5

# Image C
C_x = pymc.Uniform('C_x',37.0,40.0,value=37.990000)
covar[14,14] = 0.01
C_y = pymc.Uniform('C_y',49.8,51.8,value=50.790000)
covar[15,15] = 0.01
C_a = pymc.Uniform('C_a',20.,70.,value=44.973102)
covar[16,16] = 4.

# Image D
D_x = pymc.Uniform('D_x',78.0,80.0,value=79.020000)
covar[17,17] = 0.01
D_y = pymc.Uniform('D_y',41.0,43.0,value=42.080000)
covar[18,18] = 0.01
D_a = pymc.Uniform('D_a',15.,65.,value=33.750244)
covar[19,19] = 4.

# Image A
A_x = pymc.Uniform('A_x',87.9,91.9,value=89.880000)
covar[20,20] = 0.09
A_y = pymc.Uniform('A_y',65.9,69.9,value=67.900000)
covar[21,21] = 0.09
A_a = pymc.Uniform('A_a',50.,150.,value=109.654433)
covar[22,22] = 4.

# Image B
B_x = pymc.Uniform('B_x',77.6,81.6,value=79.600000)
covar[23,23] = 0.09
B_y = pymc.Uniform('B_y',75.9,79.9,value=77.940000)
covar[24,24] = 0.09
B_a = pymc.Uniform('B_a',5.,50.,value=24.28515)
covar[25,25] = 4.

# Galaxy
gal_q = pymc.Uniform('gal_q',0.2,1.,value=0.54)
covar[26,26] = 0.3
gal_t = pymc.Uniform('gal_t',-pi,pi,value=-0.19)
covar[27,27] = 0.5
gal_re = pymc.Uniform('gal_re',4.,80.,value=14.)
covar[28,28] = 1.
gal_a = pymc.Uniform('gal_a',0.,0.3,value=0.008)
covar[29,29] = 0.00005
gal_n = pymc.Uniform('gal_n',0.8,6.5,value=1.)
covar[30,30] = 0.5
gal_x = pymc.Uniform('gal_x',66.93,70.91,value=68.93)
gal_y = pymc.Uniform('gal_y',55.35,59.35,value=57.35)

@pymc.observed
def likelihood(value=0.,lx=lens_x,ly=lens_y,lb=lens_b,lq=lens_q,lt=lens_t,shear=shear,shear_pa=shear_pa,sx=src_x,sy=src_y,sq=src_q,st=src_t,sr=src_re,sa=src_amp,sn=src_n,Cx=C_x,Cy=C_y,Ca=C_a,Dx=D_x,Dy=D_y,Da=D_a,Ax=A_x,Ay=A_y,Aa=A_a,Bx=B_x,By=B_y,Ba=B_a,gq=gal_q,gt=gal_t,gr=gal_re,ga=gal_a,gn=gal_n,gx=gal_x,gy=gal_y):
    global pl,es,src,gal,psfFFT

    pl.x0 = lx*OVRS
    pl.y0 = ly*OVRS
    pl.b = lb*OVRS
    pl.q = lq
    pl.theta = lt

    es.x0 = lx*OVRS
    es.y0 = ly*OVRS
    es.b = shear
    es.theta = shear_pa

    src.x = sx*OVRS
    src.y = sy*OVRS
    src.q = sq
    src.theta = st
    src.re = sr*OVRS
    src.amp = sa
    src.n = sn

    gal.x = gx*OVRS
    gal.y = gy*OVRS
    gal.q = gq
    gal.theta = gt
    gal.re = gr*OVRS
    gal.amp = ga
    gal.n = 4.#gn

    srcmodel = src.eval(xc.copy(),yc.copy())
    model = pylens.lens_images([pl,es],srcmodel)
    model += gal.eval(xc.copy(),yc.copy())
    if numpy.isnan(model.sum()):
        return -1e200

    if psfFFT is None:
        psf0 = ndimage.zoom(psf,OVRS)
        psf0 /= psf0.sum()
        model,psfFFT = convolve.convolve(model,psf0)
    else:
        model,psfFFT = convolve.convolve(model,psfFFT,False)

    kernel = numpy.ones((OVRS,OVRS))
    model = signal.convolve(model,kernel,'same')[::OVRS,::OVRS]/kernel.sum()

    A = [Ax,Ay,Aa]
    B = [Bx,By,Ba]
    C = [Cx,Cy,Ca]
    D = [Dx,Dy,Da]
    for Q in [A,B,C,D]:
        Qx,Qy,Qa = Q
        xshift = Qx-int(Qx)
        yshift = Qy-int(Qy)
        qso = ndimage.shift(psf,[xshift,yshift])
        xlo = int(Qx)-qso.shape[1]/2
        xhi = xlo+qso.shape[1]
        ylo = int(Qy)-qso.shape[0]/2
        yhi = ylo+qso.shape[0]
        if xlo<0:
            qso = qso[:,-xlo:]
            xlo = 0
        if ylo<0:
            qso = qso[-ylo:,:]
            ylo = 0
        if xhi>model.shape[1]:
            off = xhi-model.shape[1]
            qso = qso[:,:-off]
            xhi = model.shape[1]
        if yhi>model.shape[0]:
            off = yhi-model.shape[0]
            qso = qso[:-off,:]
            yhi = model.shape[0]
        model[ylo:yhi,xlo:xhi] += qso*Qa

    logp = -0.5*(model-image)**2/var
    logp = logp.sum()
    if numpy.isnan(logp):
        return -1e200
    else:
        return logp


def covfromtrace(sampler,pars):
    n = sampler.trace(pars[0].__name__,0)[:].size
    vals = numpy.empty((len(pars),n))
    for i in range(len(pars)):
        name = pars[i].__name__
        vals[i] = sampler.trace(name,0)[:]
    return numpy.cov(vals)
covar /= 100000.

pars = [lens_x,lens_y,lens_b,lens_q,lens_t,shear,shear_pa,src_x,src_y,src_q,src_t,src_re,src_amp,src_n,C_x,C_y,C_a,D_x,D_y,D_a,A_x,A_y,A_a,B_x,B_y,B_a,gal_q,gal_t,gal_re,gal_a,gal_n]
pars += [gal_x,gal_y]

niter = 5e4
sampler = pymc.database.pickle.load('RUN_1')
covar = covfromtrace(sampler,pars)/10.
for item in pars:
    item.value = sampler.trace(item.__name__,0)[:][-1]


for iter in range(2):
    print "Iteration %d"%(iter+1)
    sampler = pymc.MCMC(pars+[likelihood],db='pickle',dbname='OPT.%d'%iter)
    sampler.use_step_method(MySM.MWAdaptiveMetropolis,pars,cov=covar,delay=niter*2,interval=niter*4+1,greedy=False,doLikelihood=True)
    sampler.sample(niter,niter/2)
    sampler.db.commit()
    niter *= 2
    covar = covfromtrace(sampler,pars)/10.

sampler = pymc.MCMC(pars+[likelihood],db='pickle',dbname='OPT.2')
sampler.use_step_method(MySM.MWAdaptiveMetropolis,pars,cov=covar,delay=niter+1,doLikelihood=True,Markovian=True)

sampler.sample(niter*1.2,niter*0.2)
sampler.db.commit()
