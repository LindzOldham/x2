import cPickle,numpy,pyfits as py
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
from SampleOpt import AMAOpt
import pylab as pl
import numpy as np
import myEmcee_blobs as myEmcee
#import myEmcee
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.interpolate import RectBivariateSpline

'''Update: now saves emcee output as a dictionary that's readable by the gui! This will save a load of time. Just have to shift the x,y coordinates to allow for changes in image size!'''

''' This code now also calculates the source position relative to the lens rather than relative to the origin. This means that when the lens moves, the source moves with it! I have tested this in so far as it seems to produce the same results on the final inference as before. Should maybe test it on an earlier model incarnation though.'''

# plot things
def NotPlicely(image,im,sigma,colour):
    ext = [0,image.shape[0],0,image.shape[1]]
    #vmin,vmax = numpy.amin(image), numpy.amax(image)
    pl.figure()
    pl.subplot(221)
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data')
    pl.subplot(222)
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',aspect='auto',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model')
    pl.subplot(223)
    pl.imshow(image-im,origin='lower',interpolation='nearest',extent=ext,vmin=-0.25,vmax=0.25,cmap='afmhot',aspect='auto')
    pl.colorbar()
    pl.title('data-model')
    pl.subplot(224)
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot',aspect='auto')
    pl.title('signal-to-noise residuals')
    pl.colorbar()
    py.writeto('/data/ljo31/Lens/J1144/resid'+str(colour)+'.fits',(image-im)/sigma,clobber=True)
    py.writeto('/data/ljo31/Lens/J1144/model'+str(colour)+'.fits',im,clobber=True)

    #pl.suptitle(str(V))
    #pl.savefig('/data/ljo31/Lens/TeXstuff/plotrun'+str(X)+'.png')

def SotPleparately(image,im,sigma,col):
    ext = [0,image.shape[0],0,image.shape[1]]
    pl.figure()
    pl.imshow(image,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('data - '+str(col))
    pl.figure()
    pl.imshow(im,origin='lower',interpolation='nearest',extent=ext,cmap='afmhot',vmin=0) #,vmin=vmin,vmax=vmax)
    pl.colorbar()
    pl.title('model - '+str(col))
    pl.figure()
    pl.imshow((image-im)/sigma,origin='lower',interpolation='nearest',extent=ext,vmin=-5,vmax=5,cmap='afmhot')
    pl.title('signal-to-noise residuals - '+str(col))
    pl.colorbar()

img1 = py.open('/data/ljo31/Lens/J1144/F606W_sci_cutout_biggerigger.fits')[0].data.copy()
sig1 = py.open('/data/ljo31/Lens/J1144/F606W_noisemap_biggerigger.fits')[0].data.copy()
psf1 = py.open('/data/ljo31/Lens/J1144/F606W_psf1.fits')[0].data.copy()
psf1 = psf1/np.sum(psf1)

img2 = py.open('/data/ljo31/Lens/J1144/F814W_sci_cutout_biggerigger.fits')[0].data.copy()
sig2 = py.open('/data/ljo31/Lens/J1144/F814W_noisemap_biggerigger.fits')[0].data.copy()
psf2 = py.open('/data/ljo31/Lens/J1144/F814W_psf1.fits')[0].data.copy()
psf2 = psf2/np.sum(psf2)

guiFile = '/data/ljo31/Lens/J1144/UpdatedGuiFile_23ew_iterated2'

print guiFile

result = np.load('/data/ljo31/Lens/J1144/emcee26')

lp= result[0]
print lp[:,0].shape
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
a2=0
print a1,a2,a3
trace = result[1]
dic = result[2]
print lp.shape, trace.shape

oldresult = np.load('/data/ljo31/Lens/J1144/emcee12')
olddic = oldresult[2]
oldlp=oldresult[0]
olda1,olda3 = numpy.unravel_index(oldlp[:,0].argmax(),oldlp[:,0].shape)

imgs = [img1,img2]
sigs = [sig1,sig2]
psfs = [psf1,psf2]

PSFs = []
OVRS = 2
yc,xc = iT.overSample(img1.shape,OVRS)
yo,xo = iT.overSample(img1.shape,1)
xc,xo = xc-15 , xo-15 
yc,yo = yc-10 , yo-10 
mask_I = py.open('/data/ljo31/Lens/J1144/mask_biggerigger.fits')[0].data
mask_V = py.open('/data/ljo31/Lens/J1144/mask_biggerigger.fits')[0].data
startmasks = [mask_V, mask_I]
masks,mask2s = [], []
for mask in startmasks:
    tck = RectBivariateSpline(yo[:,0],xo[0],mask)
    mask2 = tck.ev(xc,yc)
    mask2[mask2<0.5] = 0
    mask2[mask2>0.5] = 1
    mask2 = mask2==0
    mask = mask==0
    masks.append(mask)
    mask2s.append(mask2)


for i in range(len(imgs)):
    psf = psfs[i]
    image = imgs[i]
    psf /= psf.sum()
    psf = convolve.convolve(image,psf)[1]
    PSFs.append(psf)

G,L,S,offsets,shear = numpy.load(guiFile)
guishear = shear[0].copy()
pars = []
cov = []
### first parameters need to be the offsets
xoffset =  dic['xoffset'][a1,a2,a3]
yoffset = dic['yoffset'][a1,a2,a3]
pars.append(pymc.Uniform('xoffset',-5.,5.,value=xoffset))
pars.append(pymc.Uniform('yoffset',-5.,5.,value=yoffset))
cov += [0.4,0.4]

gals = []
for name in G.keys():
    s = G[name]
    p = {}
    if name == 'Galaxy 1':
        print name
        for key in 'x','y','q','pa','re','n':
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
            print lo,val,hi
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev'])
    elif name == 'Galaxy 2':
        print name
        for key in 'x','y','q','pa','re','n':
            lo,hi = s[key]['lower'],s[key]['upper']
            val = dic[name+' '+key][a1,a2,a3]
            pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
            p[key] = pars[-1]
            cov.append(s[key]['sdev'])
    gals.append(SBModels.Sersic(name,p))

lenses = []
for name in L.keys():
    s = L[name]
    p = {}
    for key in 'x','y','q','pa','b','eta':
        lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
        #val = olddic[name+' '+key][olda1,a2,olda3]
        p[key] = val
    lenses.append(MassModels.PowerLaw(name,p))
p = {}
p['x'] = lenses[0].pars['x']
p['y'] = lenses[0].pars['y']
pars.append(pymc.Uniform('extShear',-0.3,0.3,value=shear[0]['b']['value']))
cov.append(1)
p['b'] = pars[-1]
pars.append(pymc.Uniform('extShear PA',-180.,180,value=shear[0]['pa']['value']))
cov.append(100.)
p['pa'] = pars[-1]
lenses.append(MassModels.ExtShear('shear',p))


#print dic['extShear PA'][a1,a2,a3]
#print dic['extShear'][a1,a2,a3]

srcs = []
for name in S.keys():
    s = S[name]
    p = {}
    if name == 'Source 2':
        for key in 'q','re','n','pa':
           print 's1'
           lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
           val = dic[name+' '+key][a1,a2,a3]
           print lo,val,hi
           if key == 're':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.1,hi,value=val))
           elif key == 'n':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.3,hi,value=val))
           else:
               pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
           p[key] = pars[-1]
           if key == 'pa':
               cov.append(s[key]['sdev']*10) 
           elif key == 're':
               cov.append(s[key]['sdev']*1) 
           else:
               cov.append(s[key]['sdev'])
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
            print key, '= ', val
            lo,hi = lo - lenses[0].pars[key], hi - lenses[0].pars[key]
            #val = val - lenses[0].pars[key].value.item()
            print val, lo, hi
            pars.append(pymc.Uniform('%s %s'%(name,key),lo-2 ,hi+2,value=val ))   # the parameter is the offset between the source centre and the lens (in source plane obvs)
            p[key] = pars[-1] + lenses[0].pars[key] # the source is positioned at the sum of the lens position and the source offset, both of which have uniformly distributed priors.
            print p[key]
            cov.append(s[key]['sdev'])
    elif name == 'Source 1':
        print 's2'
        for key in 'q','re','n','pa':
           lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
           val = dic[name+' '+key][a1,a2,a3]
           if key == 're':
               pars.append(pymc.Uniform('%s %s'%(name,key),0.1,hi,value=val))
           else:
               pars.append(pymc.Uniform('%s %s'%(name,key),lo,hi,value=val))
           p[key] = pars[-1]
           if key == 'pa':
               cov.append(s[key]['sdev']*100) 
           elif key == 're':
               cov.append(s[key]['sdev']*10) 
           else:
               cov.append(s[key]['sdev'])
        for key in 'x','y': # subtract lens potition - to be added back on later in each likelihood iteration!
            lo,hi,val = s[key]['lower'],s[key]['upper'],s[key]['value']
            val = dic[name+' '+key][a1,a2,a3]
            print key, '= ', val
            lo,hi = lo - lenses[0].pars[key], hi - lenses[0].pars[key]
            #val = val - lenses[0].pars[key].value.item()
            print val, lo, hi
            pars.append(pymc.Uniform('%s %s'%(name,key),lo-2 ,hi+2,value=val ))   # the parameter is the offset between the source centre and the lens (in source plane obvs)
            p[key] = pars[-1] + lenses[0].pars[key] # the source is positioned at the sum of the lens position and the source offset, both of which have uniformly distributed priors.
            print p[key]
            cov.append(s[key]['sdev'])
            #p[key] = srcs[0].pars[key]
    srcs.append(SBModels.Sersic(name,p))


print len(pars), len(cov)
for p in pars:
    print p, p.value.item()

npars = []
for i in range(len(npars)):
    pars[i].value = npars[i]


trace = numpy.array(result[1])
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
for i in range(len(pars)):
    #pars[i].value = trace[a1,a2,a3,i]
    print "%18s  %8.3f"%(pars[i].__name__,pars[i].value)


colours = ['F555W', 'F814W']
models = []
fits = []
for i in range(len(imgs)):
    if i == 0:
        dx,dy = 0,0
    else:
        dx = pars[0].value 
        dy = pars[1].value 
    xp,yp = xc+dx,yc+dy
    xop,yop = xo+dy,yo+dy
    image = imgs[i]
    sigma = sigs[i]
    psf = PSFs[i]
    imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
    n = 0
    model = np.empty(((len(gals) + len(srcs)+1),imin.size))
    for gal in gals:
        print n
        gal.setPars()
        tmp = xc*0.
        tmp = gal.pixeval(xp,yp,1./OVRS,csub=1) # evaulate on the oversampled grid. OVRS = number of new pixels per old pixel.
        tmp = iT.resamp(tmp,OVRS,True) # convert it back to original size
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
    for src in srcs:
        src.setPars()
        tmp = xc*0.
        tmp = src.pixeval(x0,y0,1./OVRS,csub=1)
        tmp = iT.resamp(tmp,OVRS,True)
        tmp = convolve.convolve(tmp,psf,False)[0]
        model[n] = tmp.ravel()
        n +=1
    model[n] = np.ones(model[n-1].shape)
    print model.shape
    n +=1
    rhs = (imin/sigin) # data
    op = (model/sigin).T # model matrix
    fit, chi = optimize.nnls(op,rhs)
    print fit
    components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
    model = components.sum(0)
    models.append(model)
    NotPlicely(image,model,sigma,colours[i])
    pl.suptitle(str(colours[i]))
    pl.show()
    comps = True
    if comps == True:
        for i in range(len(gals)+len(srcs)+1):
            pl.figure()
            pl.imshow(components[i],interpolation='nearest',origin='lower',cmap='afmhot')
            pl.colorbar()
    fits.append(fit)

### show source components in the source plane
srcs1 = srcs[0].pixeval(xc,yc)*fits[0][1]
#srcs2 = srcs[1].pixeval(xc,yc)*fits[1][3]

#pl.figure()
#pl.subplot(121)
#pl.imshow(srcs1,interpolation='nearest',origin='lower',cmap='afmhot',vmin=0,vmax=0.54)
#pl.colorbar(fraction=0.046, pad=0.04)   
#pl.title('Source 2')
#pl.subplot(122)
#pl.imshow(srcs2,interpolation='nearest',origin='lower',cmap='afmhot',vmin=0,vmax=0.54)   
#pl.colorbar(fraction=0.046, pad=0.04)
#pl.title('Source 1')

dx,dy = dic['xoffset'][a1,a2,a3], dic['yoffset'][a1,a2,a3]
x1,y1,re1,n1,pa1,q1 = dic['Source 1 x'][a1,a2,a3], dic['Source 1 y'][a1,a2,a3], dic['Source 1 re'][a1,a2,a3], dic['Source 1 n'][a1,a2,a3], dic['Source 1 pa'][a1,a2,a3], dic['Source 1 q'][a1,a2,a3]
re5,n5,pa5,q5,x5,y5 = dic['Source 2 re'][a1,a2,a3], dic['Source 2 n'][a1,a2,a3], dic['Source 2 pa'][a1,a2,a3], dic['Source 2 q'][a1,a2,a3],dic['Source 2 x'][a1,a2,a3], dic['Source 2 y'][a1,a2,a3]
x2,y2,re2,n2,pa2,q2 = dic['Galaxy 1 x'][a1,a2,a3], dic['Galaxy 1 y'][a1,a2,a3], dic['Galaxy 1 re'][a1,a2,a3], dic['Galaxy 1 n'][a1,a2,a3], dic['Galaxy 1 pa'][a1,a2,a3], dic['Galaxy 1 q'][a1,a2,a3]
re3,n3,pa3,q3,x3,y3 = dic['Galaxy 2 re'][a1,a2,a3], dic['Galaxy 2 n'][a1,a2,a3], dic['Galaxy 2 pa'][a1,a2,a3], dic['Galaxy 2 q'][a1,a2,a3], dic['Galaxy 2 x'][a1,a2,a3], dic['Galaxy 2 y'][a1,a2,a3]
#if trace.shape[-1] == 26:
#    re3,n3,pa3,q3 = dic['Galaxy 2 re'][a1,a2,a3], dic['Galaxy 2 n'][a1,a2,a3], dic['Galaxy 2 pa'][a1,a2,a3], dic['Galaxy 2 q'][a1,a2,a3]
#re6,n6,pa6,q6 = dic['Galaxy 3 re'][a1,a2], dic['Galaxy 3 n'][a1,a2], dic['Galaxy 3 pa'][a1,a2], dic['Galaxy 3 q'][a1,a2]
x4,y4,b,eta,pa4,q4 = lenses[0].pars['x'],lenses[0].pars['y'],lenses[0].pars['b'],lenses[0].pars['eta'],lenses[0].pars['pa'],lenses[0].pars['q']
shear,shearpa = lenses[1].pars['b'],lenses[1].pars['pa'] 

x1,y1 = x1+lenses[0].pars['x'], y1+lenses[0].pars['y']
x5,y5=x5+x4,y5+y4
print 'source 1 ', '&', '%.2f'%x1, '&',  '%.2f'%y1, '&', '%.2f'%n1, '&', '%.2f'%re1, '&', '%.2f'%q1, '&','%.2f'%pa1,  r'\\'
print 'source 2 ', '&', '%.2f'%x5, '&',  '%.2f'%y5, '&', '%.2f'%n5, '&', '%.2f'%re5, '&', '%.2f'%q5, '&','%.2f'%pa5,  r'\\'
print 'galaxy 1 ', '&', '%.2f'%x2, '&',  '%.2f'%y2, '&', '%.2f'%n2, '&', '%.2f'%re2, '&', '%.2f'%q2, '&','%.2f'%pa2,  r'\\'
print 'galaxy 2 ', '&', '%.2f'%x3, '&',  '%.2f'%y3, '&', '%.2f'%n3, '&', '%.2f'%re3, '&', '%.2f'%q3, '&','%.2f'%pa3,  r'\\'

#if trace.shape[-1] == 26:
#    print 'source 2 ', '&', '%.2f'%x1, '&',  '%.2f'%y1, '&', '%.2f'%n5, '&', '%.2f'%re5, '&', '%.2f'%q5, '&','%.2f'%pa5,  r'\\'
    #print 'galaxy 2 ', '&', '%.2f'%x2, '&',  '%.2f'%y2, '&', '%.2f'%n3, '&', '%.2f'%re3, '&', '%.2f'%q3, '&','%.2f'%pa3,  r'\\'

print 'lens 1 ', '&', '%.2f'%x4, '&',  '%.2f'%y4, '&', '%.2f'%eta, '&', '%.2f'%b, '&', '%.2f'%q4, '&','%.2f'%pa4,  r'\\\hline'

print 'shear = ', '%.4f'%shear, 'shear pa = ', '%.2f'%shearpa

pl.figure()
pl.plot(lp[:,0,:])
'''
for name in G.keys():
    s = G[name]
    if name == 'Galaxy 1':
        for key in 'x','y','q','pa','re','n':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3])
    elif name == 'Galaxy 2':
        for key in 'q','pa','re','n':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3])
        for key in 'x','y':
            s[key]['value'] = np.array(dic['Galaxy 1 '+key][a1,a2,a3])

for name in L.keys():
    s = L[name]
    for key in 'x','y','q','pa','b','eta':
        s[key]['value'] = np.array(olddic[name+' '+key][olda1,a2,olda3])

for name in guishear.keys():
    guishear[name]['value'] = np.array(olddic['Lens 1 '+key][olda1,a2,olda3])

guishear['b']['value'] = np.array(olddic['extShear'][olda1,a2,olda3])
guishear['pa']['value'] = np.array(olddic['extShear PA'][olda1,a2,olda3])

for name in S.keys():
    s = S[name]
    if name == 'Source 1':
        for key in 'q','re','n','pa':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3])
        for key in 'x','y':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3] + olddic['Lens 1 '+key][olda1,a2,olda3]) 
    elif name == 'Source 2':
        print name
        for key in 'q','re','n','pa':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3])
        for key in 'x','y':
            s[key]['value'] = np.array(dic[name+' '+key][a1,a2,a3] + olddic['Lens 1 '+key][olda1,a2,olda3])

print guishear
offsets[0][3] = np.array(dic['xoffset'][a1,a2,a3])
offsets[1][3] = np.array(dic['yoffset'][a1,a2,a3])
cPickle.dump([G,L,S,offsets,[guishear,True]],open('/data/ljo31/Lens/J1144/UpdatedGuiFile_24','wb'))
'''
