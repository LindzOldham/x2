from linslens import EELsModels as L
import numpy as np, pylab as pl, pyfits as py

'''for name in ['J0837','J0913','J1125','J1144']:#,'J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1606','J1619','J2228']:
    print name
    result0 = np.load('/data/ljo31/Lens/LensModels/'+name+'_211')
    try:
        result1 = np.load('/data/ljo31/Lens/'+name+'/twoband_0')
    except:
        result1 = np.load('/data/ljo31/Lens/'+name+'/twoband_1')
    # old model
    model0 = L.EELs(result0,name=name)
    model0.Initialise()
    lp0 = model0.lp
    ### new model
    model1 = L.EELs(result1,name=name)
    model1.Initialise()
    lp1 = model1.lp
    print name, lp0,lp1'''


for name in ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1606','J1619','J2228']:
    #print name
    result = np.load('/data/ljo31/Lens/LensModels/'+name+'_211')

    lp,trace,dic,_= result
    a1,a3 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
    a2=0
    
    try:
        gal1,gal2,src1 = dic['Galaxy 1 re'][a1,a2,a3], dic['Galaxy 2 re'][a1,a2,a3], dic['Source 1 re'][a1,a2,a3]
        vals = [gal1,gal2,src1]
        ns = [dic['Galaxy 1 n'][a1,a2,a3], dic['Galaxy 2 n'][a1,a2,a3], dic['Source 1 n'][a1,a2,a3]]
    except:
        gal1,src1 = dic['Galaxy 1 re'][a1,a2,a3], dic['Source 1 re'][a1,a2,a3]
        vals = [gal1,src1]
        ns = [dic['Galaxy 1 n'][a1,a2,a3], dic['Source 1 n'][a1,a2,a3]]
    #print name, '%.2f'%np.max(vals), ['%.2f'%val for val in vals], ['%.2f'%n for n in ns]
    try:
        print name, '& ', '%.2f'%gal1, '&', '%.2f'%gal2, '&', '%.2f'%ns[0], '&', '%.2f'%ns[1], '&', '%.2f'%src1,'&', '%.2f'%np.max(vals), r'\\'
    except:
        print name, '& ', '%.2f'%gal1, '& -- &',  '%.2f'%ns[0], '& -- &', '%.2f'%src1,'&', '%.2f'%np.max(vals), r'\\'
