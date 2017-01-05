import numpy as np, pylab as pl, pyfits as py
from linslens import EELsModels as L
import indexTricks as iT
from tools.simple import *
from scipy import ndimage
import special_functions as sf


# may 2013
angles = [52.,-52.,-8.,-1.,96.,75.,-18.]
names = ['J1323','J1347','J1446','J1605','J1606','J1619','J2228']
midx = [70.,34.,56.,45.,80.,82,146.]
midy = [70.,37.,58.,45.,75.,88,130.]
centroid = []
source1,source2 = [],[]
def get_profile(name,angle,mx,my,rein):
    print name, angle,mx,my
    try:
        result = np.load('/data/ljo31/Lens/LensModels/twoband/'+name+'_211')
    except:
        result = np.load('/data/ljo31/Lens/LensModels/'+name+'_211')
    model = L.EELs(result,name)
    model.Initialise()
    cutout = model.img1[my-30:my+30,mx-30:mx+30]
    #pl.figure()
    #climshow(cutout)
    
    y,x = iT.coords(cutout.shape)
    line = np.arange(-30.,30.,1)
    xline, yline = -line*np.sin(angle*np.pi/180.), line*np.cos(angle*np.pi/180.)
    xline = np.array([int(xline[i]) for i in range(xline.size)])
    yline = np.array([int(yline[i]) for i in range(yline.size)])
    
    pl.figure()
    pl.subplot(211)
    pl.plot(xline,yline)
    pl.axis([-30,30,-30,30])
    slit = np.array([cutout[yline[i]+30.,xline[i]+30.] for i in range(xline.size)])
    pl.subplot(212)
    pl.plot(slit)
    pl.axvline(30.+rein,color='k')
    pl.axvline(30.-rein,color='k')
    pl.title(name)
    # now convolve to seeing of esi and find centroid. Record where source and lens should be relative to centroid. These should give the centroids of the apertures.
    # need to know seeing in arcseconds and pixel scale. Or seeing in pixels, for both.
    # hst seeing = 2 pixels fwhm - so ~ 1 pixel
    # esi seeing = 4 pixels?
    # convolve:
    disp = (4.**2 - (2./2.355)**2)**0.5
    convslit = ndimage.gaussian_filter1d(slit.copy(),disp)
    pl.plot(convslit)
    # now work out centroid offset from centre
    fit = np.array([0.,convslit.max(),convslit.argmax(),1.])
    fit = sf.ngaussfit(slit,fit)[0]
    pl.axvline(fit[2],color='k')
    print name, 30.-fit[2]
    centroid.append([name,30.-fit[2]])
    # you basically can't see where the second peak should be. So let's just put it at the Einstein radius
    # what we want to output is just: (1) given the peak of the Gaussian, where should we put the lens aperture (-centroid) and where should we put the source apertures (-centroid+rein, -centroid-rein)
    source1.append([name,rein])

# get einstein radii in hst pixels, and convert to esi pixels
table = np.load('/data/ljo31/Lens/LensParams/Structure_1src_huge_new.npy')[()][0]
rein = np.array([table[NAME]['Lens 1 b'] for NAME in names])

for kk in range(len(names)):
    get_profile(names[kk],angles[kk],midx[kk],midy[kk],rein[kk])

np.save('/data/ljo31/Lens/LensParams/esi_centroids_may2014',dict(centroid))
np.save('/data/ljo31/Lens/LensParams/esi_reins_may2014',dict(source1))

# by hand, look at image and see distance of brightest source image from centre. Put the source aperture basically here. Then try extracting. Monday work... :(

'''
name,angle='J2228',90.
try:
    result = np.load('/data/ljo31/Lens/LensModels/twoband/'+name+'_211')
except:
    result = np.load('/data/ljo31/Lens/LensModels/'+name+'_211')
model = L.EELs(result,name)
model.Initialise()
cutout = model.img1
y,x = iT.coords(cutout.shape)
line = np.arange(-np.amax(y)/2.,np.amax(y)/2.,1)
xline, yline = line*np.sin(angle*np.pi/180.), line*np.cos(angle*np.pi/180.)
xline = np.array([int(xline[i]) for i in range(xline.size)])
yline = np.array([int(yline[i]) for i in range(yline.size)])
pl.figure()
pl.subplot(211)
pl.plot(xline,yline)
slit = np.array([cutout[yline[i]+np.amax(y)/2.,xline[i]+np.amax(y)/2.] for i in range(xline.size)])
pl.subplot(212)
pl.plot(slit)
'''
