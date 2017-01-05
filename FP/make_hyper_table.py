import numpy as np
from tools import gus_plotting as g
import pylab as pl

dir = '/data/ljo31b/EELs/FP/inference/FP_infer_'
results = ['gNFW_gamma_HYPER', 'gNFW_12_HYPER','NFW_12_HYPER']

for file in results:
    print file
    result = np.load(dir+file)
    lp,trace,dic,_ = result
    a1,a2 = np.unravel_index(lp.argmax(),lp.shape)
    #med = [dic[key][a1,a3] for key in dic.keys()]
    #lo = [np.percentile(dic[key][4000:].ravel(),16) for key in dic.keys()]
    #hi = [np.percentile(dic[key][4000:].ravel(),84) for key in dic.keys()]
    for key in dic.keys():
        med = dic[key][a1,a2]
        lo = np.percentile(dic[key][4000:].ravel(),16)
        hi = np.percentile(dic[key][4000:].ravel(),84)
        print key, '%.2f'%med, '%.2f'%(med-lo), '%.2f'%(hi-med), '%.2f'%np.median(dic[key][4000:].ravel())

    # make an informative triangle plot
    chain = g.changechain(trace[4000:],filename=dir+file+'_chain.dat')
    if 'gamma' in file:
        labels = ['$\mu_{\gamma}$', r'$\tau_{\gamma}$']
    elif 'gNFW_12' in file:
        labels = ['$\mu_{\gamma}$','$\mu_{r_s}$', r'$\tau_{\gamma}$', r'$\tau_{r_s}$' ]
    else:
        labels = ['$\mu_{\Gamma}$', r'$\tau_{\Gamma}$']
    g.triangle_plot(chain, axis_labels = labels,nbins=30)
    pl.show()
