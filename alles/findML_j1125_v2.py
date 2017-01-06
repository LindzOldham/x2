import numpy as np, pylab as pl, pyfits as py
from scipy.interpolate import splrep, splint, splev
import pymc
import myEmcee_blobs as myEmcee
import cPickle
from stellarpop import tools,distances 
import indexTricks as iT
from imageSim import SBModels,convolve
from pylens import *

sz = np.load('/data/ljo31/Lens/LensParams/LensRedshifts.npy')[()]
bands = np.load('/data/ljo31/Lens/LensParams/HSTBands.npy')[()]
szs = np.load('/data/ljo31/Lens/LensParams/SourceRedshifts.npy')[()]
vi,vk,dvi,dvk,v,i,k,Re,dRe = np.load('/data/ljo31/Lens/LensParams/colours_lens_2src.npy').T[3]
vis,vks,dvis,dvks,vs,i_s,ks,Res,dRes = np.load('/data/ljo31/Lens/LensParams/colours_2src.npy').T[3]

## for the source, we need the LENSED magnitude
result = np.load('/data/ljo31/Lens/LensModels/J1125_212_nonconcentric')
from linslens import EELsModels as L
model = L.EELs(result,'J1125')
model.Initialise()
yo,xo = iT.coords(model.img1.shape)
fluxes = []
ZPs = [26.493,25.947]
for ii in range(2):
    if ii == 0:
        Dx,Dy = 0,0
    else:
        Dx,Dy = model.Ddic['xoffset'], model.Ddic['yoffset']
    xp,yp=xo+Dx,yo+Dy
    lenses = model.lenses
    for lens in lenses:
        lens.setPars()
    x0,y0 = pylens.lens_images(model.lenses,model.srcs,[xp,yp],1.,getPix=True)
    flux = xp*0.
    for jj in range(2):
        src = model.srcs[jj]
        src.setPars()
        tmp = xp*0.
        tmp = src.pixeval(x0,y0,1.,csub=23)
        tmp = iT.resamp(tmp,1,True)
        tmp = convolve.convolve(tmp,model.PSFs[ii],False)[0]
        flux += model.fits[ii][jj+2]*tmp
    mag = -2.5*np.log10(flux.sum()) + ZPs[ii]
    fluxes.append(mag)

vs, i_s = fluxes
## now for K band
kresult = np.load('/data/ljo31/Lens/LensModels/J1125_Kp_212')
from linslens import EELsKeckModels as L
model = L.EELs(kresult, result, 'J1125')
model.Initialise()
yo,xo = iT.coords(model.img.shape)*model.pix
ZP = np.load('/data/ljo31/Lens/LensParams/Keck_zeropoints.npy')[()]['J1125']
Dx,Dy = model.Dx+model.Ddic['xoffset'],model.Dy+model.Ddic['yoffset']
xp,yp=xo+Dx,yo+Dy
lenses = model.lenses
for lens in lenses:
    lens.setPars()
x0,y0 = pylens.lens_images(lenses,model.srcs,[xp,yp],model.pix,getPix=True)
flux = xp*0.
for jj in range(2):
    src = model.srcs[jj]
    src.setPars()
    tmp = xp*0.
    tmp = src.pixeval(x0,y0,model.pix,csub=23)
    tmp = iT.resamp(tmp,1,True)
    tmp = convolve.convolve(tmp,model.psf,False)[0]
    flux += model.fits[jj+2]*tmp
ks = -2.5*np.log10(flux.sum()) + ZP
print ks
vs,i_s,ks = vs - 0.047, i_s-0.029,ks-0.006 # dereddening!!

dist = distances.Distance()
dist.h = 0.7
dist.OMEGA_M = 0.3
dist.OMEGA_L = 0.7
cm_per_Mpc = 3.08568e24
chabrier = np.load('/home/mauger/python/stellarpop/chabrier.dat')
age_array = chabrier['age']
age_array = np.log10(age_array)
age_array[0]=5.
wave=chabrier['wave']
solarSpectra = chabrier[6]
## (1) : make grids
kfilt = tools.filterfromfile('Kp_NIRC2')
ifilt = tools.filterfromfile('F814W_ACS')
v6filt = tools.filterfromfile('F606W_ACS')
Ufilt = tools.filterfromfile('u_SDSS')
Gfilt = tools.filterfromfile('g_SDSS')
Rfilt = tools.filterfromfile('r_SDSS')
Ifilt = tools.filterfromfile('i_SDSS')
Zfilt = tools.filterfromfile('z_SDSS')
interps_K, interps_I, interps_V,corr = [],[],[],[]
U,G,R,I,Z = [],[],[],[],[]
zz = [sz['J1125'], szs['J1125']]
for ii in range(2):
    z = zz[ii]
    dl = dist.Dl(z)*cm_per_Mpc
    conv = 1./(4.*np.pi*dl**2.)
    conv *= 3.826e33
    corr.append(conv)
    SED = solarSpectra*conv
    grid_K,grid_I,grid_V = np.zeros(age_array.size),np.zeros(age_array.size),np.zeros(age_array.size)
    gU,gG,gR,gI,gZ = grid_K*0.,grid_K*0.,grid_K*0.,grid_K*0.,grid_K*0.
    for a in range(age_array.size):
        grid_K[a] = tools.ABFM(kfilt,[wave,SED[a]],z)
        grid_I[a] = tools.ABFM(ifilt,[wave,SED[a]],z)
        grid_V[a] = tools.ABFM(v6filt,[wave,SED[a]],z)
        gU[a],gG[a] = tools.ABFM(Ufilt,[wave,SED[a]],z), tools.ABFM(Gfilt,[wave,SED[a]],z)
        gR[a],gI[a] = tools.ABFM(Rfilt,[wave,SED[a]],z), tools.ABFM(Ifilt,[wave,SED[a]],z)
        gZ[a] = tools.ABFM(Zfilt,[wave,SED[a]],z)
    interps_K.append(splrep(age_array,grid_K))
    interps_I.append(splrep(age_array,grid_I))
    interps_V.append(splrep(age_array,grid_V))
    U.append(splrep(age_array,gU))
    G.append(splrep(age_array,gG))
    R.append(splrep(age_array,gR))
    I.append(splrep(age_array,gI))
    Z.append(splrep(age_array,gZ))


pars = []
cov = []
pars.append(pymc.Uniform('log age j1125',8,10.1,value=9.5))
pars.append(pymc.Uniform('log lens mass j1125',10,12.5,value=11.))
pars.append(pymc.Uniform('log source mass j1125',9,11.9,value=11.))
cov += [1.]
optCov=np.array(cov)
print len(pars)

u_sdss,g_sdss,r_sdss,i_sdss,z_sdss = 23.36,20.59,18.89,17.93,17.46
# deredden sdss as well
u_sdss,g_sdss,r_sdss,i_sdss,z_sdss = u_sdss-0.081,g_sdss-0.063,r_sdss-0.044,i_sdss-0.033,z_sdss-0.024
du,dg,dr,di,dz = 1.25,0.05,0.02,0.01,0.03
dgr,diz = np.sqrt(dg**2.+dr**2.), np.sqrt(di**2.+dz**2.)
dgi = np.sqrt(dg**2.+di**2.)

data  = np.array([v,vs,i,i_s,k,ks,g_sdss-r_sdss,i_sdss-z_sdss,g_sdss-i_sdss])
sigma = np.array([dvi,dvis,dvi,dvis,dvk,dvks,dgr,diz,dgi])

data  = np.array([v,vs,i,i_s,k,ks,g_sdss,r_sdss,i_sdss,z_sdss])
sigma = np.array([dvi,dvis,dvi,dvis,dvk,dvks,dg,dr,di,dz])


mu = [1,9]
@pymc.deterministic
def logP(value=0.,p=pars):
    model_i=np.zeros(2)
    model_v,model_k=model_i*0.,model_i*0.
    Fu,Fg,Fr,Fi,Fz = 0.,0.,0.,0.,0.
    logT,M = pars[0].value,[pars[1].value,pars[2].value]
    for kk in range(2):
        model_v[kk],model_i[kk] = splev(logT,interps_V[kk]).item()-2.5*M[kk],splev(logT,interps_I[kk]).item()-2.5*M[kk]
        model_k[kk] = splev(logT,interps_K[kk]).item()-2.5*M[kk]
        ##
        Fu += 10**M[kk]*10**(-0.4*splev(logT,U[kk]).item()) * mu[kk]
        Fg += 10**M[kk]*10**(-0.4*splev(logT,G[kk]).item()) * mu[kk]
        Fr += 10**M[kk]*10**(-0.4*splev(logT,R[kk]).item()) * mu[kk]
        Fi += 10**M[kk]*10**(-0.4*splev(logT,I[kk]).item()) * mu[kk]
        Fz += 10**M[kk]*10**(-0.4*splev(logT,Z[kk]).item()) * mu[kk]
    mod_u,mod_g,mod_r = -2.5*np.log10(Fu),-2.5*np.log10(Fg),-2.5*np.log10(Fr)
    mod_i,mod_z = -2.5*np.log10(Fi),-2.5*np.log10(Fz)
    gr, iz,gi = mod_g-mod_r, mod_i-mod_r,mod_g-mod_i
    #model_vi,model_vk = model_v-model_i, model_v-model_k
    #model = np.array([model_vi[0],model_vi[1],model_vk[0],model_vk[1],mod_u,mod_g,mod_r,mod_i,mod_z])
    model = np.array([model_v[0],model_v[1],model_i[0],model_i[1],model_k[0],model_k[1],gr,iz,gi])
    resid = -0.5*(data-model)**2./sigma**2.
    lp = resid.sum()
    return lp

@pymc.observed
def likelihood(value=0.,lp=logP):
    return lp

S = myEmcee.PTEmcee(pars+[likelihood],cov=optCov,nthreads=20,nwalkers=10,ntemps=3)
S.sample(5000)
#outFile = '/data/ljo31/Lens/Analysis/1src_VKIage_lensgals_physprior'
outFile = '/data/ljo31/Lens/Analysis/j1125_age_dereddened_2'#_2'
f = open(outFile,'wb')
cPickle.dump(S.result(),f,2)
f.close()
result = S.result()
lp,trace,dic,_ = result
a2=0
a1,a3 = np.unravel_index(lp[:,0].argmax(),lp[:,0].shape)
ftrace=trace[:,0].reshape((trace.shape[0]*trace.shape[2],trace.shape[3]))
for i in range(len(pars)):
    pars[i].value = np.percentile(ftrace[:,i],50,axis=0)
    print "%18s  %8.5f"%(pars[i].__name__,pars[i].value)
print 'lp = ', np.amax(lp[:,0])


model_i=np.zeros(2)
model_v,model_k=model_i*0.,model_i*0.
Fu,Fg,Fr,Fi,Fz = 0.,0.,0.,0.,0.
logT,M = pars[0].value,[pars[1].value,pars[2].value]
for kk in range(2):
    model_v[kk],model_i[kk] = splev(logT,interps_V[kk]).item()-2.5*M[kk],splev(logT,interps_I[kk]).item()-2.5*M[kk]
    model_k[kk] = splev(logT,interps_K[kk]).item()-2.5*M[kk]
    ##
    Fu += 10**M[kk]*10**(-0.4*splev(logT,U[kk]).item()) * mu[kk]
    Fg += 10**M[kk]*10**(-0.4*splev(logT,G[kk]).item()) * mu[kk]
    Fr += 10**M[kk]*10**(-0.4*splev(logT,R[kk]).item()) * mu[kk]
    Fi += 10**M[kk]*10**(-0.4*splev(logT,I[kk]).item()) * mu[kk]
    Fz += 10**M[kk]*10**(-0.4*splev(logT,Z[kk]).item()) * mu[kk]
mod_u,mod_g,mod_r = -2.5*np.log10(Fu),-2.5*np.log10(Fg),-2.5*np.log10(Fr)
mod_i,mod_z = -2.5*np.log10(Fi),-2.5*np.log10(Fz)
gr, iz = mod_g-mod_r, mod_i-mod_r
model = np.array([model_v[0],model_v[1],model_i[0],model_i[1],model_k[0],model_k[1],gr,iz])
np.save('/data/ljo31/Lens/LensParams/j1125_age_params',model)

for ii in range(data.size):
    print '%.2f'%data[ii], '%.2f'%model[ii]

'''
# also uncertainties!!!
burnin=1000
f = trace[burnin:,0].reshape((trace[burnin:].shape[0]*trace[burnin:].shape[2],trace[burnin:].shape[3]))
params = np.zeros((f.shape[0]/100.,f.shape[1]))
Mlens,Msrc,age,vis_lens,vks_lens,vis_src,vks_src = params*0.,params*0.,params*0.,params*0.,params*0.,params*0.,params*0.

ll=0
for j in range(0,len(f),100):
    logT,M1,M2 = f[j]
    M = [M1,M2]
    for kk in range(2):
        model_v[kk],model_i[kk] = splev(logT,interps_V[kk]).item()-2.5*M[kk],splev(logT,interps_I[kk]).item()-2.5*M[kk]
        model_k[kk] = splev(logT,interps_K[kk]).item()-2.5*M[kk]
        ##
        Fu += 10**M[kk]*10**(-0.4*splev(logT,U[kk]).item())
        Fg += 10**M[kk]*10**(-0.4*splev(logT,G[kk]).item())
        Fr += 10**M[kk]*10**(-0.4*splev(logT,R[kk]).item())
        Fi += 10**M[kk]*10**(-0.4*splev(logT,I[kk]).item())
        Fz += 10**M[kk]*10**(-0.4*splev(logT,Z[kk]).item())
    mod_u,mod_g,mod_r = -2.5*np.log10(Fu),-2.5*np.log10(Fg),-2.5*np.log10(Fr)
    mod_i,mod_z = -2.5*np.log10(Fi),-2.5*np.log10(Fz)
    Mlens[ll] = M1
    Msrc[ll] = M2
    age[ll] = logT
    vis_lens[ll],vks_lens[ll] = model_v[0]-model_i[0], model_v[0]-model_k[0]
    vis_src[ll],vks_src[ll] = model_v[1]-model_i[1], model_v[1]-model_k[1]
    ll +=1

data = []
dic = dict([('logTs',age),('Mlens',Mlens),('Msrc',Msrc),('vi lens',vis_lens),('vk lens',vks_lens),('vi src',vis_src),('vk src',vks_src)])
for quant in dic.keys():
    print quant
    np.save('/data/ljo31/Lens/Analysis/J1125_'+str(quant),dic[quant])
    lo,med,hi = np.percentile(dic[quant],[16,50,84],axis=0)
    data.append([lo,med,hi])
np.save('/data/ljo31/Lens/Analysis/j1125_InferredAge',np.array(data))

'''
