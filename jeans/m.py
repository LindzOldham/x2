import numpy as np, pylab as pl

result1 = np.load('/data/ljo31b/EELs/inference/FP/jeansmodels_nugg_halfre_shen_1')
result2 = np.load('/data/ljo31b/EELs/inference/FP/jeansmodels_nugg_1re_shen_1')
result3 = np.load('/data/ljo31b/EELs/inference/FP/jeansmodels_nugg_2re_shen_1')
result4 = np.load('/data/ljo31b/EELs/inference/FP/jeansmodels_nugg_1re_shen_2')
result5 = np.load('/data/ljo31b/EELs/inference/FP/jeansmodels_nugg_2re_shen_2')
dic1,dic2,dic3,dic4,dic5 = result1[2],result2[2],result3[2],result4[2],result5[2]

for key in ['a','b','alpha','sigma']:
    pl.figure()
    pl.hist(dic1[key][1000:].ravel(),30,alpha=0.5,histtype='stepfilled')
    pl.hist(dic2[key][1000:].ravel(),30,alpha=0.5,histtype='stepfilled')
    pl.hist(dic3[key][1000:].ravel(),30,alpha=0.5,histtype='stepfilled')
    #pl.hist(dic4[key][1000:].ravel(),30,alpha=0.5,histtype='stepfilled')
    pl.hist(dic5[key][1000:].ravel(),30,alpha=0.5,histtype='stepfilled')
    pl.title(key)
