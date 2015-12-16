import cPickle,numpy,pyfits
import pymc
from pylens import *
from imageSim import SBModels,convolve
import indexTricks as iT
import pylab as pl
import numpy as np

result = np.load('/data/ljo31/Lens/J1323/emcee49') # two-galaxy best model
result = np.load('/data/ljo31/Lens/J1323/emcee48') # three-galaxy best model
result = np.load('/data/ljo31/Lens/J1323/emcee53') # two-galaxy best model

lp= result[0]
a1,a2 = numpy.unravel_index(lp.argmax(),lp.shape)
trace = result[1]
det = result[2]
dx,dy = det['xoffset'][a1,a2], det['yoffset'][a1,a2]

srcs,gals,lenses = [],[],[]
srcs.append(SBModels.Sersic('Source 1', {'x':det['Source 2 x'][a1,a2],'y':det['Source 2 y'][a1,a2],'q':det['Source 1 q'][a1,a2],'pa':det['Source 1 pa'][a1,a2],'re':det['Source 1 re'][a1,a2],'n':det['Source 1 n'][a1,a2]}))
srcs.append(SBModels.Sersic('Source 2', {'x':det['Source 2 x'][a1,a2],'y':det['Source 2 y'][a1,a2],'q':det['Source 2 q'][a1,a2],'pa':det['Source 2 pa'][a1,a2],'re':det['Source 2 re'][a1,a2],'n':det['Source 2 n'][a1,a2]}))
gals.append(SBModels.Sersic('Galaxy 1', {'x':det['Galaxy 1 x'][a1,a2],'y':det['Galaxy 1 y'][a1,a2],'q':det['Galaxy 1 q'][a1,a2],'pa':det['Galaxy 1 pa'][a1,a2],'re':det['Galaxy 1 re'][a1,a2],'n':det['Galaxy 1 n'][a1,a2]}))
gals.append(SBModels.Sersic('Galaxy 2', {'x':det['Galaxy 1 x'][a1,a2],'y':det['Galaxy 1 y'][a1,a2],'q':det['Galaxy 2 q'][a1,a2],'pa':det['Galaxy 2 pa'][a1,a2],'re':det['Galaxy 2 re'][a1,a2],'n':det['Galaxy 2 n'][a1,a2]}))
if trace.shape[-1] == 34:
    gals.append(SBModels.Sersic('Galaxy 3', {'x':det['Galaxy 1 x'][a1,a2],'y':det['Galaxy 1 y'][a1,a2],'q':det['Galaxy 3 q'][a1,a2],'pa':det['Galaxy 3 pa'][a1,a2],'re':det['Galaxy 3 re'][a1,a2],'n':det['Galaxy 3 n'][a1,a2]}))
lenses.append(MassModels.PowerLaw('Lens 1', {'x':det['Lens 1 x'][a1,a2],'y':det['Lens 1 y'][a1,a2],'q':det['Lens 1 q'][a1,a2],'pa':det['Lens 1 pa'][a1,a2],'b':det['Lens 1 b'][a1,a2],'eta':det['Lens 1 eta'][a1,a2]}))
lenses.append(MassModels.ExtShear('shear',{'x':det['Lens 1 x'][a1,a2],'y':det['Lens 1 y'][a1,a2],'b':det['extShear'][a1,a2], 'pa':det['extShear PA'][a1,a2]}))

print 'Source 1 ',
for key in ['Source 2 x', 'Source 2 y', 'Source 1 n', 'Source 1 re', 'Source 1 q', 'Source 1 pa']:
    lower, mid, upper = np.percentile(det[key][200:],[16,50,84])
    maxL = det[key][a1,a2]
    lo,up = maxL-lower, upper-maxL
    print '& $%.2f'%maxL, '_{-','%.2f'%lo, '}^{+','%.2f'%up, '}$',
print r'\\'
print 'Source 2 ', 
for key in ['Source 2 x', 'Source 2 y', 'Source 2 n', 'Source 2 re', 'Source 2 q', 'Source 2 pa']:
    lower, mid, upper = np.percentile(det[key][200:],[16,50,84])
    maxL = det[key][a1,a2]
    lo,up = maxL-lower, upper-maxL
    print '& $%.2f'%maxL, '_{-','%.2f'%lo, '}^{+','%.2f'%up, '}$',
print r'\\'
print 'Galaxy 1',
for key in ['Galaxy 1 x', 'Galaxy 1 y', 'Galaxy 1 n', 'Galaxy 1 re', 'Galaxy 1 q', 'Galaxy 1 pa']:
    lower, mid, upper = np.percentile(det[key][200:],[16,50,84])
    maxL = det[key][a1,a2]
    lo,up = maxL-lower, upper-maxL
    print '& $%.2f'%maxL, '_{-','%.2f'%lo, '}^{+','%.2f'%up, '}$' ,
print r'\\'
print 'Galaxy 2',
for key in ['Galaxy 1 x', 'Galaxy 1 y', 'Galaxy 2 n', 'Galaxy 2 re', 'Galaxy 2 q', 'Galaxy 2 pa']:
    lower, mid, upper = np.percentile(det[key][200:],[16,50,84])
    maxL = det[key][a1,a2]
    lo,up = maxL-lower, upper-maxL
    print '& $%.2f'%maxL, '_{-','%.2f'%lo, '}^{+','%.2f'%up, '}$',
print r'\\'
if trace.shape[-1] == 34:
    print 'Galaxy 3',
    for key in ['Galaxy 1 x', 'Galaxy 1 y', 'Galaxy 3 n', 'Galaxy 3 re', 'Galaxy 3 q', 'Galaxy 3 pa']:
        lower, mid, upper = np.percentile(det[key][000:],[16,50,84])
        maxL = det[key][a1,a2]
        lo,up = maxL-lower, upper-maxL
        print '& $%.2f'%maxL, '_{-','%.2f'%lo, '}^{+','%.2f'%up, '}$',
    print r'\\'
print 'Lens 1',
for key in ['Lens 1 x', 'Lens 1 y', 'Lens 1 eta', 'Lens 1 b', 'Lens 1 q', 'Lens 1 pa']:
    lower, mid, upper = np.percentile(det[key][200:],[16,50,84])
    maxL = det[key][a1,a2]
    lo,up = maxL-lower, upper-maxL
    print '& $%.2f'%maxL, '_{-','%.2f'%lo, '}^{+','%.2f'%up, '}$',
print r'\\\hline'
print 'Shear',
lower, mid, upper = np.percentile(det['extShear'][200:],[16,50,84])
maxL = det['extShear'][a1,a2]
lo,up = maxL-lower, upper-maxL
print '& $%.2f'%maxL, '_{-','%.2f'%lo, '}^{+','%.2f'%up, '}$ & & & & &', r'\\'
print 'shear PA',
lower, mid, upper = np.percentile(det['extShear PA'][200:],[16,50,84])
maxL = det['extShear PA'][a1,a2]
lo,up = maxL-lower, upper-maxL
print '& $%.2f'%maxL, '_{-','%.2f'%lo, '}^{+','%.2f'%up, '}$ & & & & & ', r'\\\hline'

## save uncertainties to a file, as they will probably be useful laterz
