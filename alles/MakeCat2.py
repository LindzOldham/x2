import EELsModels as L
import numpy as np

def MakeTab(model):
    model.MakeDict()
    model.BuildLenses()
    model.BuildGalaxies()
    model.BuildSources()
    model.EasyAddImages()
    model.GetFits(plotresid=False)
    mags = model.GetIntrinsicMags()
    Res = model.GetSourceSize(kpc=True)
    restmags, Ls = model.GetPhotometry()
    mus = model.GetSB()
    restmus = model.GetRestSB()
    model.MakePDFDict()
    model.GetPDFs()
    med,lo,hi = model.UncertaintiesFromPDF(makecat=True)
    return mags,mus,restmus,restmags,Res,Ls,med,lo,hi


cats_m, cats_l, cats_h = [], [], []
result = np.load('/data/ljo31/Lens/LensModels/J0837_211')
model = L.EELs(result,name='J0837')
print 'J0837'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)
print len(hi)

result = np.load('/data/ljo31/Lens/LensModels/J0901_211')
model = L.EELs(result,name='J0901')
print 'J0901'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J0913_211')
model = L.EELs(result,name='J0913')
print 'J0913'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1125_211')
model = L.EELs(result,name='J1125')
print 'J1125'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1144_211')
model = L.EELs(result,name='J1144')
print 'J1144'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)




result = np.load('/data/ljo31/Lens/LensModels/J1218_211')
model = L.EELs(result,name='J1218')
print 'J1218'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1323_211') 
model = L.EELs(result,name='J1323')
print 'J1323'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)

result = np.load('/data/ljo31/Lens/LensModels/J1347_211')
model = L.EELs(result,name='J1347')
print 'J1347'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1446_211')
model = L.EELs(result,name='J1446')
print 'J1446'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)

result = np.load('/data/ljo31/Lens/LensModels/J1605_211') 
model = L.EELs(result,name='J1605')
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1606_211')
model = L.EELs(result,name='J1606')
print 'J1606'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J2228_211')
model = L.EELs(result,name='J2228')
print 'J2228'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)

print len(cats_m), cats_m[0], cats_m

cats_m, cats_l, cats_h = np.array(cats_m), np.array(cats_l), np.array(cats_h)


names = ['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1606','J2228']
types = [('s1','S16'),('mag v',float),('mag i',float),('rest mag v',float),('rest mag i',float),('mu v',float),('mu i',float),('rest mu v',float),('rest mu i',float),('Re v',float),('Re i',float),('lum v',float),('lum i',float),('lum b',float),('rest mag b',float),('mag v lo',float),('mag i lo',float),('rest mag v lo',float),('rest mag i lo',float),('mu v lo',float),('mu i lo',float),('rest mu v lo',float),('rest mu i lo',float),('Re v lo',float),('Re i lo',float),('lum v lo',float),('lum i lo',float),('lum b lo',float),('rest mag b lo',float),('mag v hi',float),('mag i hi',float),('rest mag v hi',float),('rest mag i hi',float),('mu v hi',float),('mu i hi',float),('rest mu v hi',float),('rest mu i hi',float),('Re v hi',float),('Re i hi',float),('lum v hi',float),('lum i hi',float),('lum b hi',float),('rest mag b hi',float)]
data=np.array(zip(names,cats_m[:,0],cats_m[:,1],cats_m[:,2],cats_m[:,3],cats_m[:,4],cats_m[:,5],cats_m[:,6],cats_m[:,7],cats_m[:,8],cats_m[:,9],cats_m[:,10],cats_m[:,11],cats_m[:,12],cats_m[:,13],cats_l[:,0],cats_l[:,1],cats_l[:,2],cats_l[:,3],cats_l[:,4],cats_l[:,5],cats_l[:,6],cats_l[:,7],cats_l[:,8],cats_l[:,9],cats_l[:,10],cats_l[:,11],cats_l[:,12],cats_l[:,13],cats_h[:,0],cats_h[:,1],cats_h[:,2],cats_h[:,3],cats_h[:,4],cats_h[:,5],cats_h[:,6],cats_h[:,7],cats_h[:,8],cats_h[:,9],cats_h[:,10],cats_h[:,11],cats_h[:,12],cats_h[:,13]),dtype=types)

np.save('/data/ljo31/Lens/LensParams/PhotCat_1src',data)
np.savetxt('/data/ljo31/Lens/LensParams/PhotCat_1src.txt',data,fmt=["%s"] +["%.3f",]*14*3)


## 2 src models
cats_m, cats_l, cats_h = [], [], []


#result = np.load('/data/ljo31/Lens/LensModels/J0913_212_concentric')
#model = L.EELs(result,name='J0913')
#print 'J0913 concentric'
#MakeTab(model)

result = np.load('/data/ljo31/Lens/LensModels/J0913_212_nonconcentric')
model = L.EELs(result,name='J0913')
print 'J0913 nonconcentric'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)

#result = np.load('/data/ljo31/Lens/LensModels/J1125_212_concentric')
#model = L.EELs(result,name='J1125')
#print 'J1125 concentric'
#MakeTab(model)

result = np.load('/data/ljo31/Lens/LensModels/J1125_212_nonconcentric')
model = L.EELs(result,name='J1125')
print 'J1125 nonconcentric'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1144_212_allparams')
model = L.EELs(result,name='J1144')
print 'J1144'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1323_212') 
model = L.EELs(result,name='J1323')
print 'J1323'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1347_112')
model = L.EELs(result,name='J1347')
print 'J1347'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1446_212')
model = L.EELs(result,name='J1446')
print 'J1446'
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1605_212_final') 
model = L.EELs(result,name='J1605')
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1606_112')
model = L.EELs(result,name='J1606')
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)

result = np.load('/data/ljo31/Lens/LensModels/J2228_212')
model = L.EELs(result,name='J2228')
mag,mu,restmu,restmag,Re,l,med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


cats_m, cats_l, cats_h = np.array(cats_m), np.array(cats_l), np.array(cats_h)

types = [('s1','S16'),('mag v',float),('mag i',float),('rest mag v',float),('rest mag i',float),('mu v',float),('mu i',float),('rest mu v',float),('rest mu i',float),('Re v',float),('Re i',float),('lum v',float),('lum i',float),('lum b',float),('rest mag b',float),('mag v lo',float),('mag i lo',float),('rest mag v lo',float),('rest mag i lo',float),('mu v lo',float),('mu i lo',float),('rest mu v lo',float),('rest mu i lo',float),('Re v lo',float),('Re i lo',float),('lum v lo',float),('lum i lo',float),('lum b lo',float),('rest mag b lo',float),('mag v hi',float),('mag i hi',float),('rest mag v hi',float),('rest mag i hi',float),('mu v hi',float),('mu i hi',float),('rest mu v hi',float),('rest mu i hi',float),('Re v hi',float),('Re i hi',float),('lum v hi',float),('lum i hi',float),('lum b hi',float),('rest mag b hi',float)]
names = ['J0913','J1125','J1144','J1323','J1347','J1446','J1605','J1606','J2228']
data=np.array(zip(names,cats_m[:,0],cats_m[:,1],cats_m[:,2],cats_m[:,3],cats_m[:,4],cats_m[:,5],cats_m[:,6],cats_m[:,7],cats_m[:,8],cats_m[:,9],cats_m[:,10],cats_m[:,11],cats_m[:,12],cats_m[:,13],cats_l[:,0],cats_l[:,1],cats_l[:,2],cats_l[:,3],cats_l[:,4],cats_l[:,5],cats_l[:,6],cats_l[:,7],cats_l[:,8],cats_l[:,9],cats_l[:,10],cats_l[:,11],cats_l[:,12],cats_l[:,13],cats_h[:,0],cats_h[:,1],cats_h[:,2],cats_h[:,3],cats_h[:,4],cats_h[:,5],cats_h[:,6],cats_h[:,7],cats_h[:,8],cats_h[:,9],cats_h[:,10],cats_h[:,11],cats_h[:,12],cats_h[:,13]),dtype=types)



np.save('/data/ljo31/Lens/LensParams/PhotCat_2src',data)
np.savetxt('/data/ljo31/Lens/LensParams/PhotCat_2src.txt',data,fmt=["%s"] +["%.3f"]*14*3)
