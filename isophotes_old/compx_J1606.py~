import numpy as np, pylab as pl, pyfits as py
from linslens import EELsModels as L
from tools.simple import printn, climshow
from scipy.interpolate import splrep, splev,splint
import indexTricks as iT
from scipy import ndimage
from tools.fitEllipse import *
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from numpy import cos, sin

result = np.load('/data/ljo31/Lens/LensModels/J1605_212_final')
name='J1605'
from linslens import EELsModels as L
model = L.EELs(result,name)
model.Initialise()
print model.GetIntrinsicMags()
model.GetSourceSize(kpc=True)
z=model.z
Da=model.Da
scale=model.scale
srcs=model.srcs
imgs=model.imgs
fits=model.fits

models = model.models
comps = model.components
imgs = model.imgs
yo,xo = iT.overSample(imgs[0].shape,1)
#xo -= srcs[0].pars['x']
#yo -= srcs[0].pars['y']

S1 = fits[0][-3]*srcs[0].pixeval(xo,yo,1.,csub=31)
S2 = fits[0][-2]*srcs[1].pixeval(xo,yo,1.,csub=31)


# ok. We want to make contours, and fit said contours with ellipses.
fracs = [0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99,0.999]

def ellipse(H,xbins,ybins,smooth=[1,1],color='CornflowerBlue'):
    # compute contours, which we can then fit ellipses to
    #H,xbins,ybins = pl.histogram2d(ax,ay,bins=[xbins,ybins])
    #pl.figure()
    #pl.imshow(H,interpolation='nearest',origin='lower')
    #H = ndimage.gaussian_filter(H,smooth)
    sortH = np.sort(H.flatten())
    cumH = sortH.cumsum()
    # 1, 2, 3-sigma, for the old school:
    lvl00 = 2*sortH.max()
    def lvl(frac):
        return sortH[cumH>cumH.max()*frac].min()
    # extract contours
    pl.figure()
    cn = pl.contour(H,[lvl(frac) for frac in fracs],colors=color,\
                  extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
    pl.axhline(0)
    pl.axvline(0)
    
    
    p=[]
    v = []
    x,y = [],[]
    print len(cn.collections)
    pl.figure()
    for i in range(len(fracs)-4):
        print i
        p.append(cn.collections[i].get_paths()[0])
        v.append(p[-1].vertices)
        X,Y = v[-1].T
        x.append(X)
        y.append(Y)
    #return x[0],y[0]

    '''# now we want to fit each ellipse
    pl.figure()
    for i in range(len(fracs)-2):
        pl.plot(x[i],y[i])'''

    # now we want to fit each ellipse so that it actually works
    for i in range(len(y)):
        X,Y = x[i],y[i]
        a = pymc.Uniform('a',0,50-3*i)
        b = pymc.Uniform('b',0,50-3*i)
        alpha = pymc.Uniform('alpha',-0.1,1.1)
        pars = [a,b,alpha]
        cov=np.array([1.,1.,0.2])
    
        
        @pymc.deterministic
        def logP(value=0.,p=pars):
            A = (X*cos(alpha.value) + Y*sin(alpha.value))/a.value
            B = (X*sin(alpha.value) - Y*cos(alpha.value))/b.value
            eq = A**2. + B**2. - 1.
            return np.sum(-eq**2.)
                

        @pymc.observed
        def likelihood(value=0.,lp=logP):
            return lp

        S = myEmcee.Emcee(pars+[likelihood],cov=cov,nthreads=1,nwalkers=100)
        S.sample(2000)
        outFile = '/data/ljo31/Lens/compx_'+name+'_'+str(fracs[i])
        f = open(outFile,'wb')
        cPickle.dump(S.result(),f,2)
        f.close()
        result = S.result()
        lp,trace,dic,_= result
        a2,a3 = np.unravel_index(lp.argmax(),lp.shape)
        for j in range(len(pars)):
            pars[j].value = trace[a2,a3,j]
            print "%18s  %8.3f"%(pars[j].__name__,pars[j].value)

        
        # plot ellipse fit
        R = np.arange(0,2.*np.pi, 0.01)
        xx = a.value*cos(R)*cos(alpha.value) - b.value*sin(R)*sin(alpha.value)
        yy = a.value*cos(R)*sin(alpha.value) + b.value*sin(R)*cos(alpha.value)
        #pl.figure()
        #pl.plot(X,Y,'b')
        #pl.plot(xx,yy, color = 'red')
        
        pl.figure()
        pl.subplot(321)
        pl.plot(lp)
        pl.subplot(322)
        pl.plot(dic['a'])
        pl.title('a')
        pl.subplot(323)
        pl.plot(dic['b'])
        pl.title('b')
        pl.subplot(324)
        pl.plot(dic['alpha'])
        pl.title('alpha')
                
        # plot ellipse fit
        R = np.arange(0,2.*np.pi, 0.01)
        xx = a.value*cos(R)*cos(alpha.value) - b.value*sin(R)*sin(alpha.value)
        yy = a.value*cos(R)*sin(alpha.value) + b.value*sin(R)*cos(alpha.value)
        pl.subplot(325)
        pl.plot(X,Y)
        pl.plot(xx,yy, color = 'red')


x,y = xo[0],yo[:,0]
x -= srcs[0].pars['x']
y -= srcs[0].pars['y']

#x,y=x-95.,y-100.

print np.mean(x),np.mean(y)
xbins = [x[i]+0.5*(x[i+1]-x[i]) for i in range(len(x)-1)]
ybins = [y[i]+0.5*(y[i+1]-y[i]) for i in range(len(y)-1)]
xbins=np.array(xbins)
ybins=np.array(ybins)

ellipse(S1,xbins,ybins)
pl.show()
