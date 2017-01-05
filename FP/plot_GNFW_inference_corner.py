import numpy as np, pylab as pl, pyfits as py
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from tools import solarmag
from stellarpop import distances
from SampleOpt import AMAOpt
from astLib import astCalc
from scipy.misc import logsumexp

# ready to receive better chains!

#outFile = '/data/ljo31b/EELs/FP/inference/FP_infer_gNFW_12_HYPER_LO_M_salpeter_BIG'
#result = np.load(outFile)
#lp,trace,dic,_ = result
#a1,a2 = np.unravel_index(lp.argmax(),lp.shape)

outFile = '/data/ljo31b/EELs/FP/inference/FP_infer_gNFW_12_HYPER_LO_M_salpeter_BIG_PT'
result = np.load(outFile)
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)


#pl.figure()
#pl.plot(lp[2000:])

bins = np.arange(0.3,1.8,0.05)

ch1,ch2 = dic['gamma'][15000:,0].ravel(), dic['r0'][15000:,0].ravel()
ch = np.column_stack((ch1,ch2))
from corner_plot import corner_plot, multi_corner_plot

corner_plot(ch,axis_labels=[r'$\gamma$','$r_0$'],nbins=10)
pl.show()


outFile = '/data/ljo31b/EELs/FP/inference/FP_infer_gNFW_12_HYPER_LO_M'
result = np.load(outFile)
lp,trace,dic,_ = result
a1,a2 = np.unravel_index(lp.argmax(),lp.shape)

ch1,ch2 = dic['gamma'][10000:].ravel(), dic['r0'][10000:].ravel()
CH = np.column_stack((ch1,ch2))

#corner_plot(CH,axis_labels=[r'$\gamma$','$r_0$'],nbins=10)
#pl.show()

multi_corner_plot([ch,CH],axis_labels=[r'$\gamma$','$r_0$ / kpc'],chain_labels=['Salpeter','Chabrier'],linecolors=['SteelBlue','Red'],linewidth=3,nbins=12)
pl.show()
