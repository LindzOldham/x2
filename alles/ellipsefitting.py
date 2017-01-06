import numpy as np, pylab as pl, pyfits as py, cPickle
from tools.fitEllipse import *
from scipy import ndimage

def clip(arr,nsig=4.5):
    a = np.sort(arr.ravel())
    a = a[a.size*0.001:a.size*0.999]
    while 1:
        m,s,l = a.mean(),a.std(),a.size
        a = a[abs(a-m)<nsig*s]
        if a.size==l:
            return m,s

def clip2d(arr1,arr2):
    m1,s1 = clip(arr1)
    m2,s2 = clip(arr2)
    ii=np.where((abs(arr1-m1)<=2*s1) & (abs(arr2-m2)<=2*s2))
    return ii

## practise on data with enough points in it!
def ellipse(ax,ay,xbins,ybins,smooth=[4,2],color='CornflowerBlue'):
    # compute contours, which we can then fit ellipses to
    H,xbins,ybins = pl.histogram2d(ax,ay,bins=[xbins,ybins])
    #pl.figure()
    #pl.imshow(H,interpolation='nearest',origin='lower')
    H = ndimage.gaussian_filter(H,smooth)
    sortH = np.sort(H.flatten())
    cumH = sortH.cumsum()
    # 1, 2, 3-sigma, for the old school:
    lvl00 = 2*sortH.max()
    lvl68 = sortH[cumH>cumH.max()*0.32].min()
    lvl95 = sortH[cumH>cumH.max()*0.05].min()
    lvl99 = sortH[cumH>cumH.max()*0.003].min()
    # extract contours
    pl.figure()
    cn = pl.contour(H,[lvl68,lvl95],colors=color,\
                  extent=(xbins[0],xbins[-1],ybins[0],ybins[-1]))
    p = cn.collections[0].get_paths()[0]
    v=p.vertices
    x,y=v.T
    a = fitEllipse(x,y)
    sxy,sx2,sy2 = a[1]/2.,a[0],a[2]
    sx,sy=abs(sx2)**0.5,abs(sy2)**0.5
    rho = sxy/sx/sy
    axes = ellipse_axis_length(a)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation2(a)
    a, b = axes
    R = np.linspace(-np.pi,np.pi,100)
    xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
    yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
    pl.plot(xx,yy)
    return rho, sx,sy,sxy


name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']
res,masses = np.load('/data/ljo31/Lens/Analysis/ReMassTrace_199.npy')
sigmas = np.zeros((name.size,4))
for ii in range(name.size):
    pl.title(name[ii-1])
    jj = clip2d(res[ii],masses[ii])
    x,y = np.log10(res[ii,jj])[0], masses[ii,jj][0]
    xbins=np.linspace(min(x)-0.5,max(x)+0.5,100)
    ybins=np.linspace(min(y)-0.1,max(y)+0.1,100)
    if name[ii]=='J0913':
        sigmas[ii] = ellipse(x,y,xbins,ybins,smooth=[3,2])
    elif name[ii] == 'J0837':
        sigmas[ii] = ellipse(x,y,xbins,ybins,smooth=[2,2])
    else:
        sigmas[ii] = ellipse(x,y,xbins,ybins)
    
np.save('/data/ljo31/Lens/LensParams/ReMass_covariances.npy',sigmas)



'''
### old - for saving the trace file ###
masses = np.load('/data/ljo31/Lens/Analysis/ReMassTrace_200_asigo.npy')[:,:199]
res = np.zeros((12,199))
name = py.open('/data/ljo31/Lens/LensParams/Phot_1src.fits')[1].data['name']
for ii in range(name.size):
    Res,mags = np.load('/data/ljo31/Lens/Analysis/ReMagPDFs_'+str(name[ii])+'.npy')
    rev,rei = Res.T
    res[ii] = rev[:199]

np.save('/data/ljo31/Lens/Analysis/ReMassTrace_199',[res,masses])
'''
