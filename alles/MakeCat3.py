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
    model.GetPDFs(kpc=True)
    med,lo,hi = model.UncertaintiesFromPDF(makecat=True)
    return med,lo,hi


cats_m, cats_l, cats_h = [], [], []
result = np.load('/data/ljo31/Lens/LensModels/J0837_211')
model = L.EELs(result,name='J0837')
print 'J0837'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)
print len(hi)



result = np.load('/data/ljo31/Lens/LensModels/J0901_211')
model = L.EELs(result,name='J0901')
print 'J0901'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J0913_211')
model = L.EELs(result,name='J0913')
print 'J0913'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1125_211')
model = L.EELs(result,name='J1125')
print 'J1125'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1144_211')
model = L.EELs(result,name='J1144')
print 'J1144'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)




result = np.load('/data/ljo31/Lens/LensModels/J1218_211')
model = L.EELs(result,name='J1218')
print 'J1218'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1323_211') 
model = L.EELs(result,name='J1323')
print 'J1323'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)

result = np.load('/data/ljo31/Lens/LensModels/J1347_211')
model = L.EELs(result,name='J1347')
print 'J1347'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1446_211')
model = L.EELs(result,name='J1446')
print 'J1446'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)

result = np.load('/data/ljo31/Lens/LensModels/J1605_211') 
model = L.EELs(result,name='J1605')
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1606_211')
model = L.EELs(result,name='J1606')
print 'J1606'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J2228_211')
model = L.EELs(result,name='J2228')
print 'J2228'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


m, l, h = np.array(cats_m), np.array(cats_l), np.array(cats_h)


from astropy.io.fits import *
names = np.array(['J0837','J0901','J0913','J1125','J1144','J1218','J1323','J1347','J1446','J1605','J1606','J2228'])
c1 = Column(name='name', format='A5',array=names)
c2 = Column(name='mag v', format='D',array=m[:,0])
c3 = Column(name='mag i', format='D',array=m[:,1])
c4 = Column(name='rest mag v', format='D',array=m[:,3])
c5 = Column(name='rest mag i', format='D',array=m[:,4])
c6 = Column(name='rest mag b', format='D',array=m[:,5])
c7 = Column(name='mu v', format='D',array=m[:,6])
c8 = Column(name='mu i', format='D',array=m[:,7])
c9 = Column(name='rest mu v', format='D',array=m[:,9])
c10 = Column(name='rest mu i', format='D',array=m[:,10])
c11 = Column(name='rest mu b', format='D',array=m[:,11])
c12 = Column(name='Re v', format='D',array=m[:,12])
c13 = Column(name='Re i', format='D',array=m[:,13])
c14 = Column(name='lum v', format='D',array=m[:,15])
c15 = Column(name='lum i', format='D',array=m[:,16])
c16 = Column(name='lum b', format='D',array=m[:,17])

### and uncertainties - lower bounds
c2l = Column(name='mag v lo', format='D',array=l[:,0])
c3l = Column(name='mag i lo', format='D',array=l[:,1])
c4l = Column(name='rest mag v lo', format='D',array=l[:,3])
c5l = Column(name='rest mag i lo', format='D',array=l[:,4])
c6l = Column(name='rest mag b lo', format='D',array=l[:,5])
c7l = Column(name='mu v lo', format='D',array=l[:,6])
c8l = Column(name='mu i lo', format='D',array=l[:,7])
c9l = Column(name='rest mu v lo', format='D',array=l[:,9])
c10l = Column(name='rest mu i lo', format='D',array=l[:,10])
c11l = Column(name='rest mu b lo', format='D',array=l[:,11])
c12l = Column(name='Re v lo', format='D',array=l[:,12])
c13l = Column(name='Re i lo', format='D',array=l[:,13])
c14l = Column(name='lum v lo', format='D',array=l[:,15])
c15l = Column(name='lum i lo', format='D',array=l[:,16])
c16l = Column(name='lum b lo', format='D',array=l[:,17])


### and upper bounds
c2h = Column(name='mag v hi', format='D',array=h[:,0])
c3h = Column(name='mag i hi', format='D',array=h[:,1])
c4h = Column(name='rest mag v hi', format='D',array=h[:,3])
c5h = Column(name='rest mag i hi', format='D',array=h[:,4])
c6h = Column(name='rest mag b hi', format='D',array=h[:,5])
c7h = Column(name='mu v hi' , format='D',array=h[:,6])
c8h = Column(name='mu i hi', format='D',array=h[:,7])
c9h = Column(name='rest mu v hi', format='D',array=h[:,9])
c10h = Column(name='rest mu i hi', format='D',array=h[:,10])
c11h = Column(name='rest mu b hi', format='D',array=h[:,11])
c12h = Column(name='Re v hi', format='D',array=h[:,12])
c13h = Column(name='Re i hi', format='D',array=h[:,13])
c14h = Column(name='lum v hi', format='D',array=h[:,15])
c15h = Column(name='lum i hi', format='D',array=h[:,16])
c16h = Column(name='lum b hi', format='D',array=h[:,17])



coldefs = ColDefs([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c2l,c3l,c4l,c5l,c6l,c7l,c8l,c9l,c10l,c11l,c12l,c13l,c14l,c15l,c16l,c2h,c3h,c4h,c5h,c6h,c7h,c8h,c9h,c10h,c11h,c12h,c13h,c14h,c15h,c16h])
tbhdu = BinTableHDU.from_columns(coldefs)
tbhdu.writeto('/data/ljo31/Lens/LensParams/Phot_1src.fits',clobber=True)


## 2 src models
cats_m, cats_l, cats_h = [], [], []


#result = np.load('/data/ljo31/Lens/LensModels/J0913_212_concentric')
#model = L.EELs(result,name='J0913')
#print 'J0913 concentric'
#MakeTab(model)

result = np.load('/data/ljo31/Lens/LensModels/J0913_212_nonconcentric')
model = L.EELs(result,name='J0913')
print 'J0913 nonconcentric'
med,lo,hi = MakeTab(model)
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
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1144_212_allparams')
model = L.EELs(result,name='J1144')
print 'J1144'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1323_212') 
model = L.EELs(result,name='J1323')
print 'J1323'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1347_112')
model = L.EELs(result,name='J1347')
print 'J1347'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1446_212')
model = L.EELs(result,name='J1446')
print 'J1446'
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


result = np.load('/data/ljo31/Lens/LensModels/J1605_212_final') 
model = L.EELs(result,name='J1605')
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)



result = np.load('/data/ljo31/Lens/LensModels/J1606_112')
model = L.EELs(result,name='J1606')
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)

result = np.load('/data/ljo31/Lens/LensModels/J2228_212')
model = L.EELs(result,name='J2228')
med,lo,hi = MakeTab(model)
cats_m.append(med)
cats_l.append(lo)
cats_h.append(hi)


m, l, h = np.array(cats_m), np.array(cats_l), np.array(cats_h)


from astropy.io.fits import *
names = np.array(['J0913','J1125','J1144','J1323','J1347','J1446','J1605','J1606','J2228'])
c1 = Column(name='name', format='A5',array=names)
c2 = Column(name='mag v', format='D',array=m[:,0])
c3 = Column(name='mag i', format='D',array=m[:,1])
c4 = Column(name='rest mag v', format='D',array=m[:,3])
c5 = Column(name='rest mag i', format='D',array=m[:,4])
c6 = Column(name='rest mag b', format='D',array=m[:,5])
c7 = Column(name='mu v', format='D',array=m[:,6])
c8 = Column(name='mu i', format='D',array=m[:,7])
c9 = Column(name='rest mu v', format='D',array=m[:,9])
c10 = Column(name='rest mu i', format='D',array=m[:,10])
c11 = Column(name='rest mu b', format='D',array=m[:,11])
c12 = Column(name='Re v', format='D',array=m[:,12])
c13 = Column(name='Re i', format='D',array=m[:,13])
c14 = Column(name='lum v', format='D',array=m[:,15])
c15 = Column(name='lum i', format='D',array=m[:,16])
c16 = Column(name='lum b', format='D',array=m[:,17])

### and uncertainties - lower bounds
c2l = Column(name='mag v lo', format='D',array=l[:,0])
c3l = Column(name='mag i lo', format='D',array=l[:,1])
c4l = Column(name='rest mag v lo', format='D',array=l[:,3])
c5l = Column(name='rest mag i lo', format='D',array=l[:,4])
c6l = Column(name='rest mag b lo', format='D',array=l[:,5])
c7l = Column(name='mu v lo', format='D',array=l[:,6])
c8l = Column(name='mu i lo', format='D',array=l[:,7])
c9l = Column(name='rest mu v lo', format='D',array=l[:,9])
c10l = Column(name='rest mu i lo', format='D',array=l[:,10])
c11l = Column(name='rest mu b lo', format='D',array=l[:,11])
c12l = Column(name='Re v lo', format='D',array=l[:,12])
c13l = Column(name='Re i lo', format='D',array=l[:,13])
c14l = Column(name='lum v lo', format='D',array=l[:,15])
c15l = Column(name='lum i lo', format='D',array=l[:,16])
c16l = Column(name='lum b lo', format='D',array=l[:,17])


### and upper bounds
c2h = Column(name='mag v hi', format='D',array=h[:,0])
c3h = Column(name='mag i hi', format='D',array=h[:,1])
c4h = Column(name='rest mag v hi', format='D',array=h[:,3])
c5h = Column(name='rest mag i hi', format='D',array=h[:,4])
c6h = Column(name='rest mag b hi', format='D',array=h[:,5])
c7h = Column(name='mu v hi' , format='D',array=h[:,6])
c8h = Column(name='mu i hi', format='D',array=h[:,7])
c9h = Column(name='rest mu v hi', format='D',array=h[:,9])
c10h = Column(name='rest mu i hi', format='D',array=h[:,10])
c11h = Column(name='rest mu b hi', format='D',array=h[:,11])
c12h = Column(name='Re v hi', format='D',array=h[:,12])
c13h = Column(name='Re i hi', format='D',array=h[:,13])
c14h = Column(name='lum v hi', format='D',array=h[:,15])
c15h = Column(name='lum i hi', format='D',array=h[:,16])
c16h = Column(name='lum b hi', format='D',array=h[:,17])



coldefs = ColDefs([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c2l,c3l,c4l,c5l,c6l,c7l,c8l,c9l,c10l,c11l,c12l,c13l,c14l,c15l,c16l,c2h,c3h,c4h,c5h,c6h,c7h,c8h,c9h,c10h,c11h,c12h,c13h,c14h,c15h,c16h])
tbhdu = BinTableHDU.from_columns(coldefs)
tbhdu.writeto('/data/ljo31/Lens/LensParams/Phot_2src.fits',clobber=True)

