import glob,cPickle,numpy
from spsmodels import zspsmodel as spsmodel
from scipy import interpolate
import sys,time
import numpy as np

# Grab the SED models
files = glob.glob('/data/mauger/STELLARPOP/chabrier_Z*dat')
axes = {'Z':[],'age':None,'tau':None,'tau_V':None}

for file in files:
    z = float(file.split('=')[1].split('.dat')[0])
    axes['Z'].append(z)
axes['Z'] = numpy.sort(numpy.unique(numpy.asarray(axes['Z'])))

# We could open any of the files; they'll all have the same tau, tau_V, age
#   arrays.
f = open(file)
tmp = cPickle.load(f) # Trash read of output
del tmp
tmp = cPickle.load(f) # Trash read of wave
del tmp
t = cPickle.load(f)
tV = cPickle.load(f)
a = cPickle.load(f)
f.close()

axes['age'] = a
axes['tau'] = t
axes['tau_V'] = tV

# The interpolation order of the input data. Use linear (order=1) for the
#   crazy 'Z' sampling chosen by BC03.
order = {'Z':1,'age':3,'tau':3,'tau_V':3}
axes_models = {}
interpmodels = {}

for key in axes.keys():
    arr = axes[key]
    axes_models[key] = {}
    axes_models[key]['points'] = numpy.sort(arr)
    axes_models[key]['eval'] = interpolate.splrep(arr,numpy.arange(arr.size),k=order[key],s=0)


filters = ['B_Johnson','I_Cousins','V_Johnson','F606W_ACS']
redshift = 0.64

outname = '/data/ljo31b/EELs/spsmodels/wide/BIV_F606W_z05_chabBC03.model'
f = open(outname,'wb')

axes_models['redshift'] = {'points':numpy.array([redshift])}
print "Creating model: ",outname
t = time.time()
model = spsmodel.zSPSModel(files,axes_models,filters)

cPickle.dump(model,f,2)
f.close()
print "Time elapsed: ",(time.time()-t)

