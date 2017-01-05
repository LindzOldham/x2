import pyfits as py, numpy as np
from imageSim import SBModels,convolve
from pylens import *
import indexTricks as iT
from scipy import optimize
from scipy.interpolate import splrep, splev, splint
from astLib import astCalc
from linslens.GetPhotometry import *
import itertools
from linslens.Plotter import *

zs, zl = 1.165, 0.581
# get dust corrections

class Lens:
    def __init__(self,result,fits=None):
        self.result = result
        self.lp, self.trace, self.dic,_ = self.result
        self.fits = fits
        if self.trace.shape[1] < 10:
            self.lp, self.trace =self.lp[:,0], self.trace[:,0]
            d = []
            for key in self.dic.keys():
                d.append((key,self.dic[key][:,0]))
            self.dic = dict(d)
        self.dic['Source 2 x'] -= self.dic['Lens 1 x']
        self.dic['Source 2 y'] -= self.dic['Lens 1 y']

    def MakeDict(self):
        if 'Source 2 re' not in self.dic.keys():
            self.srcno = 1
        else:
            self.srcno = 2
        if 'Galaxy 2 re' not in self.dic.keys():
            self.galno = 1
        else:
            self.galno = 2

    
        l,u,d = [], [], []
        for key in self.dic.keys():
            f = self.dic[key].reshape((self.trace.shape[0]*self.trace.shape[1]))
            lo,med,up = np.percentile(f,50)-np.percentile(f,16), np.percentile(f,50), np.percentile(f,84)-np.percentile(f,50) 
            d.append((key,med))
            l.append((key,lo))
            u.append((key,up))
        
        if 'Source 1 x' not in self.dic.keys():
            for key in 'x', 'y':
                f = self.dic['Source 2 '+key].reshape((self.trace.shape[0]*self.trace.shape[1]))
                lo,med,up = np.percentile(f,50)-np.percentile(f,16), np.percentile(f,50), np.percentile(f,84)-np.percentile(f,50) 
                d.append(('Source 1 '+key,med))
                l.append(('Source 1 '+key,lo))
                u.append(('Source 1 '+key,up))
                self.dic['Source 1 '+key] = self.dic['Source 2 '+key]
        if 'Galaxy 2 x' not in self.dic.keys():
            for key in 'x', 'y':
                f = self.dic['Galaxy 1 '+key].reshape((self.trace.shape[0]*self.trace.shape[1]))
                lo,med,up = np.percentile(f,50)-np.percentile(f,16), np.percentile(f,50), np.percentile(f,84)-np.percentile(f,50) 
                d.append(('Galaxy 2 '+key,med))
                l.append(('Galaxy 2 '+key,lo))
                u.append(('Galaxy 2 '+key,up))
                self.dic['Galaxy 2 '+key] = self.dic['Galaxy 1 '+key]
        

        self.Ddic = dict(d)                    
        self.Ldic = dict(l)
        self.Udic = dict(u)

    def PrintTable(self):
        print r'\begin{table}[H]'
        print r'\centering'
        print r'\begin{tabular}{|c|cccccc|}\hline'
        print r' object & x & y & re & n & pa & q \\\hline'
        print 'source 1 & $', '%.2f'%(self.Ddic['Source 1 x']+self.Ddic['Lens 1 x']), '_{-', '%.2f'%self.Ldic['Source 1 x'],'}^{+','%.2f'%self.Udic['Source 1 x'], '}$ & $', '%.2f'%(self.Ddic['Source 1 y']+self.Ddic['Lens 1 y']),'_{-', '%.2f'%self.Ldic['Source 1 y'],'}^{+', '%.2f'%self.Udic['Source 1 y'], '}$ & $', '%.2f'%self.Ddic['Source 1 re'],'_{-', '%.2f'%self.Ldic['Source 1 re'],'}^{+', '%.2f'%self.Udic['Source 1 re'], '}$ & $', '%.2f'%self.Ddic['Source 1 n'],'_{-', '%.2f'%self.Ldic['Source 1 n'],'}^{+', '%.2f'%self.Udic['Source 1 n'], '}$ & $','%.2f'%self.Ddic['Source 1 pa'],'_{-', '%.2f'%self.Ldic['Source 1 pa'],'}^{+', '%.2f'%self.Udic['Source 1 pa'], '}$ & $','%.2f'%self.Ddic['Source 1 q'],'_{-', '%.2f'%self.Ldic['Source 1 q'],'}^{+', '%.2f'%self.Udic['Source 1 q'], '}$',r'\\'
        ###
        if self.srcno ==2:
            print 'source 2 & $', '%.2f'%(self.Ddic['Source 2 x']+self.Ddic['Lens 1 x']), '_{-', '%.2f'%self.Ldic['Source 2 x'],'}^{+','%.2f'%self.Udic['Source 2 x'], '}$ & $', '%.2f'%(self.Ddic['Source 2 y']+self.Ddic['Lens 1 y']),'_{-', '%.2f'%self.Ldic['Source 2 y'],'}^{+', '%.2f'%self.Udic['Source 2 y'], '}$ & $', '%.2f'%self.Ddic['Source 2 re'],'_{-', '%.2f'%self.Ldic['Source 2 re'],'}^{+', '%.2f'%self.Udic['Source 2 re'], '}$ & $', '%.2f'%self.Ddic['Source 2 n'],'_{-', '%.2f'%self.Ldic['Source 2 n'],'}^{+', '%.2f'%self.Udic['Source 2 n'], '}$ & $','%.2f'%self.Ddic['Source 2 pa'],'_{-', '%.2f'%self.Ldic['Source 2 pa'],'}^{+', '%.2f'%self.Udic['Source 2 pa'], '}$ & $','%.2f'%self.Ddic['Source 2 q'],'_{-', '%.2f'%self.Ldic['Source 2 q'],'}^{+', '%.2f'%self.Udic['Source 2 q'], '}$',r'\\'
###
        print 'galaxy 1 & $', '%.2f'%self.Ddic['Galaxy 1 x'], '_{-', '%.2f'%self.Ldic['Galaxy 1 x'],'}^{+','%.2f'%self.Udic['Galaxy 1 x'], '}$ & $', '%.2f'%self.Ddic['Galaxy 1 y'],'_{-', '%.2f'%self.Ldic['Galaxy 1 y'],'}^{+', '%.2f'%self.Udic['Galaxy 1 y'], '}$ & $', '%.2f'%self.Ddic['Galaxy 1 re'],'_{-', '%.2f'%self.Ldic['Galaxy 1 re'],'}^{+', '%.2f'%self.Udic['Galaxy 1 re'], '}$ & $', '%.2f'%self.Ddic['Galaxy 1 n'],'_{-', '%.2f'%self.Ldic['Galaxy 1 n'],'}^{+', '%.2f'%self.Udic['Galaxy 1 n'], '}$ & $','%.2f'%self.Ddic['Galaxy 1 pa'],'_{-', '%.2f'%self.Ldic['Galaxy 1 pa'],'}^{+', '%.2f'%self.Udic['Galaxy 1 pa'], '}$ & $','%.2f'%self.Ddic['Galaxy 1 q'],'_{-', '%.2f'%self.Ldic['Galaxy 1 q'],'}^{+', '%.2f'%self.Udic['Galaxy 1 q'], '}$',r'\\'
        ###
        if self.galno == 2:
            print 'galaxy 2 & $', '%.2f'%self.Ddic['Galaxy 2 x'], '_{-', '%.2f'%self.Ldic['Galaxy 2 x'],'}^{+','%.2f'%self.Udic['Galaxy 2 x'], '}$ & $', '%.2f'%self.Ddic['Galaxy 2 y'],'_{-', '%.2f'%self.Ldic['Galaxy 2 y'],'}^{+', '%.2f'%self.Udic['Galaxy 2 y'], '}$ & $', '%.2f'%self.Ddic['Galaxy 2 re'],'_{-', '%.2f'%self.Ldic['Galaxy 2 re'],'}^{+', '%.2f'%self.Udic['Galaxy 2 re'], '}$ & $', '%.2f'%self.Ddic['Galaxy 2 n'],'_{-', '%.2f'%self.Ldic['Galaxy 2 n'],'}^{+', '%.2f'%self.Udic['Galaxy 2 n'], '}$ & $','%.2f'%self.Ddic['Galaxy 2 pa'],'_{-', '%.2f'%self.Ldic['Galaxy 2 pa'],'}^{+', '%.2f'%self.Udic['Galaxy 2 pa'], '}$ & $','%.2f'%self.Ddic['Galaxy 2 q'],'_{-', '%.2f'%self.Ldic['Galaxy 2 q'],'}^{+', '%.2f'%self.Udic['Galaxy 2 q'], '}$',r'\\'
        ###
        print 'lens 1 & $', '%.2f'%self.Ddic['Lens 1 x'], '_{-', '%.2f'%self.Ldic['Lens 1 x'],'}^{+','%.2f'%self.Udic['Lens 1 x'], '}$ & $', '%.2f'%self.Ddic['Lens 1 y'],'_{-', '%.2f'%self.Ldic['Lens 1 y'],'}^{+', '%.2f'%self.Udic['Lens 1 y'], '}$ & $', '%.2f'%self.Ddic['Lens 1 b'],'_{-', '%.2f'%self.Ldic['Lens 1 b'],'}^{+', '%.2f'%self.Udic['Lens 1 b'], '}$ & $', '%.2f'%self.Ddic['Lens 1 eta'],'_{-', '%.2f'%self.Ldic['Lens 1 eta'],'}^{+', '%.2f'%self.Udic['Lens 1 eta'], '}$ & $','%.2f'%self.Ddic['Lens 1 pa'],'_{-', '%.2f'%self.Ldic['Lens 1 pa'],'}^{+', '%.2f'%self.Udic['Lens 1 pa'], '}$ & $','%.2f'%self.Ddic['Lens 1 q'],'_{-', '%.2f'%self.Ldic['Lens 1 q'],'}^{+', '%.2f'%self.Udic['Lens 1 q'], '}$',r'\\\hline'
        ###
        print r'\end{tabular}'
        print r'\caption{', 'shear = $', '%.2f'%self.Ddic['extShear'], '_{-', '%.2f'%self.Ldic['extShear'],'}^{+','%.2f'%self.Udic['extShear'], '}$ , shear pa = $',  '%.2f'%self.Ddic['extShear PA'], '_{-', '%.2f'%self.Ldic['extShear PA'],'}^{+','%.2f'%self.Udic['extShear PA'], '}$}'
        print r'\end{table}'


    def BuildSources(self):
        self.srcs = []
        for number in range(1,1+self.srcno):
            name = 'Source '+str(number)
            p = {}
            for key in 'q','re','n','pa':
                p[key] = self.Ddic[name+' '+key]
            for key in 'x','y': 
                p[key] = self.Ddic[name+' '+key]+self.Ddic['Lens 1 '+key]
            self.srcs.append(SBModels.Sersic(name,p))


    def BuildGalaxies(self):
        self.gals = []
        for number in range(1,1+self.galno):
            name = 'Galaxy '+str(number)
            p = {}
            for key in 'x','y','q','re','n','pa':
                p[key] = self.Ddic[name+' '+key]
            self.gals.append(SBModels.Sersic(name,p))

    def BuildLenses(self):
        self.lenses = []
        p = {}
        for key in 'x','y','q','pa','b','eta':
            p[key] = self.Ddic['Lens 1 '+key]
        self.lenses.append(MassModels.PowerLaw('Lens 1',p))
        p = {}
        p['x'] = self.lenses[0].pars['x']
        p['y'] = self.lenses[0].pars['y']
        p['b'] = self.Ddic['extShear']
        p['pa'] = self.Ddic['extShear PA']
        self.lenses.append(MassModels.ExtShear('shear',p))

    
    def AddImages(self,img1,sig1,psf1,img2,sig2,psf2,img3,sig3,psf3,Dx=None,Dy=None):
        self.img1,self.img2,self.img3=img1,img2,img3
        self.sig1,self.sig2,self.sig3=sig1,sig2,sig3
        self.psf1,self.psf2,self.psf3=psf1,psf2,psf3
        self.Dx,self.Dy=Dx,Dy
        self.imgs = [self.img1,self.img2,self.img3]
        self.sigs = [self.sig1,self.sig2,self.sig3]
        self.psfs = [self.psf1,self.psf2,self.psf3]
        self.PSFs = []
        for i in range(len(self.imgs)):
            psf = self.psfs[i]
            image = self.imgs[i]
            psf /= psf.sum()
            psf = convolve.convolve(image,psf)[1]
            self.PSFs.append(psf)


    def EasyAddImages(self):
        ### g
        self.img1 = py.open('/data/ljo31b/lenses/chip5/imgg.fits')[0].data[450:-450,420:-480]
        bg=np.median(self.img1[-10:,-10:])
        self.img1-=bg
        self.sig1 = py.open('/data/ljo31b/lenses/chip5/noisemap_g_big.fits')[0].data
        self.psf1 = py.open('/data/ljo31b/lenses/g_psf3.fits')[0].data 
        ### r
        self.img2 = py.open('/data/ljo31b/lenses/chip5/imgr.fits')[0].data[450:-450,420:-480]
        bg=np.median(self.img2[-10:,-10:])
        self.img2-=bg
        self.sig2 = py.open('/data/ljo31b/lenses/chip5/noisemap_r_big.fits')[0].data
        self.psf2 = py.open('/data/ljo31b/lenses/r_psf3.fits')[0].data
        ### i
        self.img3 = py.open('/data/ljo31b/lenses/chip5/imgi.fits')[0].data[450:-450,420:-480]
        bg=np.median(self.img3[-10:,-10:])
        self.img3-=bg
        self.sig3 = py.open('/data/ljo31b/lenses/chip5/noisemap_i_big.fits')[0].data
        self.psf3 = py.open('/data/ljo31b/lenses/i_psf3.fits')[0].data
        ## mask
        maskg = py.open('/data/ljo31b/lenses/chip5/newmask.fits')[0].data
        maski = py.open('/data/ljo31b/lenses/chip5/mask_i.fits')[0].data
        self.mask = np.where((maskg==1)|(maski==1),1,0)
        self.mask=self.mask==0
        # offsets and OVRS
        self.Dx,self.Dy = -10.,-10.
        self.imgs = [self.img1,self.img2,self.img3]
        self.sigs = [self.sig1,self.sig2,self.sig3]
        self.psfs = [self.psf1,self.psf2,self.psf3]
        self.PSFs = []
        for i in range(len(self.imgs)):
            psf = self.psfs[i]
            image = self.imgs[i]
            psf /= psf.sum()
            psf = convolve.convolve(image,psf)[1]
            self.PSFs.append(psf)


    def GetFits(self,plotsep=False,plotresid=False,plotcomps=False,cmap='afmhot'):
        OVRS=1
        yo,xo = iT.coords(self.img1.shape)
        yc,xc=iT.overSample(self.img1.shape,OVRS)
        colours = ['g','r','i']
        models = []
        fits = []
        lp = []
        for i in range(len(self.imgs)):
            if i == 0:
                dx,dy = 0,0
            elif i == 1:
                dx = self.Ddic['gr xoffset']
                dy = self.Ddic['gr yoffset']
            elif i == 2:
                dx = self.Ddic['gi xoffset']
                dy = self.Ddic['gi yoffset']
            xp,yp = xc+dx+self.Dx,yc+dy+self.Dy
            xop,yop = xo+dx+self.Dx,yo+dy+self.Dy
            image = self.imgs[i]
            sigma = self.sigs[i]
            psf = self.PSFs[i]
            imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
            n = 0
            model = np.empty(((len(self.gals) + len(self.srcs)+1),imin.size))
            for gal in self.gals:
                gal.setPars()
                tmp = xc*0.
                tmp = gal.pixeval(xp,yp,1./OVRS,csub=23) 
                tmp = iT.resamp(tmp,OVRS,True) 
                tmp = convolve.convolve(tmp,psf,False)[0]
                model[n] = tmp.ravel()
                n +=1
            for lens in self.lenses:
                lens.setPars()
            x0,y0 = pylens.lens_images(self.lenses,self.srcs,[xp,yp],1./OVRS,getPix=True)
            for src in self.srcs:
                src.setPars()
                tmp = xc*0.
                tmp = src.pixeval(x0,y0,1./OVRS,csub=23)
                tmp = iT.resamp(tmp,OVRS,True)
                tmp = convolve.convolve(tmp,psf,False)[0]
                model[n] = tmp.ravel()
                n +=1
            model[n] = np.ones(model[n].shape)
            n +=1
            # mask
            rhs = image[self.mask]/sigma[self.mask]
            mmodel = model.reshape((n,image.shape[0],image.shape[1]))
            mmmodel = np.empty(((len(self.gals) + len(self.srcs)+1),image[self.mask].size))
            for m in range(mmodel.shape[0]):
                mmmodel[m] = mmodel[m][self.mask]
            op = (mmmodel/sigma[self.mask]).T
            # continue as normal with fit
            fit, chi = optimize.nnls(op,rhs)
            components = (model.T*fit).T.reshape((n,image.shape[0],image.shape[1]))
            model = components.sum(0)
            models.append(model)
            resid = (model.flatten()-imin)/sigin
            lp.append(-0.5*(resid**2.).sum())
            if plotsep:
                SotPleparately(image,model,sigma,' ',cmap=cmap)
            if plotresid:
                NotPlicely(image,model,sigma,colours[i],cmap=cmap)
            if plotcomps:
                CotSomponents(components,colours[i])
            fits.append(fit)
        self.fits = fits
        self.lp = lp
        self.models = models
        self.components = components
    
    def Initialise(self):
        self.MakeDict()
        self.BuildSources()
        self.BuildGalaxies()
        self.BuildLenses()
        self.EasyAddImages()
        self.GetFits(plotresid=False)

    def GetSourceMags(self): 
        self.ZPs = [25.296,25.374,25.379]
        if self.srcno == 1.:
            self.magg = self.srcs[0].getMag(self.fits[0][-2],self.ZPs[0])
            self.magr = self.srcs[0].getMag(self.fits[1][-2],self.ZPs[1])
            self.magi = self.srcs[0].getMag(self.fits[2][-2],self.ZPs[2])
        elif self.srcno == 2.:
            if np.any(self.fits[0][-3:-1]==0):
                ii = np.where(self.fits[0][-3:-1] !=0)[0]
                self.magg = self.srcs[ii].getMag(self.fits[0][ii],self.ZPs[0])
            else:
                 mg1,mg2 = self.srcs[0].getMag(self.fits[0][-3],self.ZPs[0]), self.srcs[1].getMag(self.fits[0][-2],self.ZPs[0])
                 Fg = 10**(-0.4*mg1) + 10**(-0.4*mg2)
                 self.magg =  -2.5*np.log10(Fg)
            if np.any(self.fits[1][-3:-1]==0):
                ii = np.where(self.fits[1][-3:-1] !=0)[0]
                self.magr = self.srcs[ii].getMag(self.fits[1][ii],self.ZPs[1])
            else:
                 mr1,mr2 = self.srcs[0].getMag(self.fits[1][-3],self.ZPs[1]), self.srcs[1].getMag(self.fits[1][-2],self.ZPs[1])
                 Fr = 10**(-0.4*mr1) + 10**(-0.4*mr2)
                 self.magr =  -2.5*np.log10(Fr)
            if np.any(self.fits[2][-3:-1]==0):
                ii = np.where(self.fits[2][-3:-1] !=0)[0]
                self.magi = self.srcs[ii].getMag(self.fits[2][ii],self.ZPs[2])
            else:
                 mi1,mi2 = self.srcs[0].getMag(self.fits[2][-3],self.ZPs[2]), self.srcs[1].getMag(self.fits[2][-2],self.ZPs[2])
                 Fi = 10**(-0.4*mi1) + 10**(-0.4*mi2)
                 self.magi =  -2.5*np.log10(Fi)
        return [self.magg, self.magr, self.magi]

    def GetLensMags(self):
        self.ZPs = [25.296,25.374,25.379]
        if self.galno == 1.:
            self.lensmagg = self.gals[0].getMag(self.fits[0][0],self.ZPs[0])
            self.lensmagr = self.gals[0].getMag(self.fits[1][0],self.ZPs[1])
            self.lensmagi = self.gals[0].getMag(self.fits[2][0],self.ZPs[2])
        elif self.galno == 2.:
            if np.any(self.fits[0][0:2]==0):
                ii = np.where(self.fits[0][0:2] !=0)[0]
                self.lensmagg = self.gals[ii].getMag(self.fits[0][ii],self.ZPs[0])
            else:
                 mg1,mg2 = self.gals[0].getMag(self.fits[0][0],self.ZPs[0]), self.gals[1].getMag(self.fits[0][1],self.ZPs[0])
                 Fg = 10**(-0.4*mg1) + 10**(-0.4*mg2)
                 self.lensmagg =  -2.5*np.log10(Fg)
            if np.any(self.fits[1][0:2]==0):
                ii = np.where(self.fits[1][0:2] !=0)[0]
                self.lensmagr = self.gals[ii].getMag(self.fits[1][ii],self.ZPs[1])
            else:
                 mr1,mr2 = self.gals[0].getMag(self.fits[1][0],self.ZPs[1]), self.gals[1].getMag(self.fits[1][1],self.ZPs[1])
                 Fr = 10**(-0.4*mr1) + 10**(-0.4*mr2)
                 self.lensmagr =  -2.5*np.log10(Fr)
            if np.any(self.fits[2][0:2]==0):
                ii = np.where(self.fits[2][0:2] !=0)[0]
                self.lensmagi = self.gals[ii].getMag(self.fits[2][ii],self.ZPs[2])
            else:
                 mi1,mi2 = self.gals[0].getMag(self.fits[2][0],self.ZPs[2]), self.gals[1].getMag(self.fits[2][1],self.ZPs[2])
                 Fi = 10**(-0.4*mi1) + 10**(-0.4*mi2)
                 self.lensmagi =  -2.5*np.log10(Fi)
        return [self.lensmagg, self.lensmagr, self.lensmagi]


    def GetSourceSize(self,kpc=False):
        # should probably do this with ellipses!
        self.Da = astCalc.da(zs)
        self.scale = self.Da*1e3*np.pi/180./3600.
        if self.srcno == 1:
            self.Reg = self.Ddic['Source 1 re']*0.263
            self.Rer, self.Rei = self.Reg.copy(), self.Reg.copy()
            self.Re_lower = self.Ldic['Source 1 re']*0.263
            self.Re_upper = self.Udic['Source 1 re']*0.263           
        elif self.srcno == 2:
            Xgrid = np.logspace(-4,5,1501)
            Res = []
            for i in range(len(self.imgs)):
                source = self.fits[i][-3]*self.srcs[0].eval(Xgrid) + self.fits[i][-2]*self.srcs[1].eval(Xgrid)
                R = Xgrid.copy()
                light = source*2.*np.pi*R
                mod = splrep(R,light,t=np.logspace(-3.8,4.8,1301))
                intlight = np.zeros(len(R))
                for i in range(len(R)):
                    intlight[i] = splint(0,R[i],mod)
                model = splrep(intlight[:-500],R[:-500])
                reff = splev(0.5*intlight[-1],model)
                Res.append(reff*0.263)
            self.Reg,self.Rer, self.Rei = Res
        if kpc:
            return [self.Reg*self.scale, self.Rer*self.scale, self.Rer*self.scale]
        return [self.Reg, self.Rer, self.Rei]

    def GetLensSize(self,kpc=False):
        # should probably do this with ellipses!
        self.lensDa = astCalc.da(zl)
        self.lensscale = self.lensDa*1e3*np.pi/180./3600.
        if self.galno == 1:
            self.lensReg = self.Ddic['Galaxy 1 re']*0.263
            self.lensRer, self.lensRei = self.lensReg.copy(), self.lensReg.copy()
            self.lensRe_lower = self.Ldic['Galaxy 1 re']*0.263
            self.lensRe_upper = self.Udic['Galaxy 1 re']*0.263           
        elif self.galno == 2:
            Xgrid = np.logspace(-4,5,1501)
            Res = []
            for i in range(len(self.imgs)):
                galaxy = self.fits[i][0]*self.gals[0].eval(Xgrid) + self.fits[i][1]*self.gals[1].eval(Xgrid)
                R = Xgrid.copy()
                light = galaxy*2.*np.pi*R
                mod = splrep(R,light,t=np.logspace(-3.8,4.8,1301))
                intlight = np.zeros(len(R))
                for i in range(len(R)):
                    intlight[i] = splint(0,R[i],mod)
                model = splrep(intlight[:-500],R[:-500])
                reff = splev(0.5*intlight[-1],model)
                Res.append(reff*0.263)
            self.lensReg,self.lensRer, self.lensRei = Res
        if kpc:
            return [self.lensReg*self.lensscale, self.lensRer*self.lensscale, self.lensRer*self.lensscale]
        return [self.lensReg, self.lensRer, self.lensRei]

    def GetSourceSB(self):
        self.mug = self.magg + 2.5*np.log10(2.*np.pi*self.Reg**2.)
        self.mur = self.magr + 2.5*np.log10(2.*np.pi*self.Rer**2.)
        self.mui = self.magi + 2.5*np.log10(2.*np.pi*self.Rei**2.)
        return [self.mug, self.mur, self.mui]

    def GetLensSB(self):
        self.lensmug = self.lensmagg + 2.5*np.log10(2.*np.pi*self.lensReg**2.)
        self.lensmur = self.lensmagr + 2.5*np.log10(2.*np.pi*self.lensRer**2.)
        self.lensmui = self.lensmagi + 2.5*np.log10(2.*np.pi*self.lensRei**2.)
        return [self.lensmug, self.lensmur, self.lensmui]

    def MakePDFDict(self):
        self.dictionaries = []
        for b1,b2 in itertools.product(range(self.trace.shape[0]), range(self.trace.shape[1])):
            go = []
            for key in self.dic.keys():
                go.append((key,self.dic[key][b1,b2]))
            go = dict(go)
            self.dictionaries.append(go)

    def GetPDFs(self,kpc=False):
        self.muPDF = []
        self.RePDF = []
        self.magPDF = []
        self.fitPDF = []
        self.grPDF = []
        self.giPDF = []
        OVRS=1
        yo,xo = iT.coords(self.img1.shape)
        yc,xc=iT.overSample(self.img1.shape,OVRS)
        colours = ['g','r','i']
        # start iterating
        for b1 in range(0,len(self.dictionaries),50):
            srcs,gals,lenses = [],[],[]
            # sources
            for number in range(1,1+self.srcno):
                name = 'Source '+str(number)
                p = {}
                for key in 'q','re','n','pa':
                    p[key] = self.dictionaries[b1][name+' '+key]
                for key in 'x','y':
                    p[key] = self.dictionaries[b1][name+' '+key]+self.dictionaries[b1]['Lens 1 '+key]
                srcs.append(SBModels.Sersic(name,p))
            # galaxies
            for number in range(1,1+self.galno):
                name = 'Galaxy '+str(number)
                p = {}
                for key in 'x','y','q','re','n','pa':
                    p[key] = self.dictionaries[b1][name+' '+key]
                gals.append(SBModels.Sersic(name,p))
            # lens + shear
            p = {}
            for key in 'x','y','q','pa','b','eta':
                p[key] = self.dictionaries[b1]['Lens 1 '+key]
            lenses.append(MassModels.PowerLaw('Lens 1',p))
            p = {}
            p['x'] = lenses[0].pars['x']
            p['y'] = lenses[0].pars['y']
            p['b'] = self.dictionaries[b1]['extShear']
            p['pa'] = self.dictionaries[b1]['extShear PA']
            lenses.append(MassModels.ExtShear('shear',p))
            # fits
            fits = []
            for i in range(len(self.imgs)):
                if i == 0:
                    dx,dy = 0,0
                elif i == 1.:
                    dx = self.dictionaries[b1]['gr xoffset']
                    dy = self.dictionaries[b1]['gr yoffset']
                elif i == 2.:
                    dx = self.dictionaries[b1]['gi xoffset']
                    dy = self.dictionaries[b1]['gi yoffset']
                xp,yp = xc+dx+self.Dx,yc+dy+self.Dy
                xop,yop = xo+dy+self.Dx,yo+dy+self.Dy
                image,sigma,psf = self.imgs[i],self.sigs[i],self.PSFs[i]
                imin,sigin,xin,yin = image.flatten(), sigma.flatten(),xp.flatten(),yp.flatten()
                n = 0
                model = np.empty(((len(gals) + len(srcs)+1),imin.size))
                for gal in gals:
                    gal.setPars()
                    tmp = xc*0.
                    tmp = gal.pixeval(xp,yp,1./OVRS,csub=21) 
                    tmp = iT.resamp(tmp,OVRS,True) 
                    tmp = convolve.convolve(tmp,psf,False)[0]
                    model[n] = tmp.ravel()
                    n +=1
                for lens in lenses:
                    lens.setPars()
                    x0,y0 = pylens.lens_images(lenses,srcs,[xp,yp],1./OVRS,getPix=True)
                for src in srcs:
                    src.setPars()
                    tmp = xc*0.
                    tmp = src.pixeval(x0,y0,1./OVRS,csub=21)
                    tmp = iT.resamp(tmp,OVRS,True)
                    tmp = convolve.convolve(tmp,psf,False)[0]
                    model[n] = tmp.ravel()
                    n +=1
                model[n] = np.ones(model[n].shape)
                n +=1
                # mask
                rhs = image[self.mask]/sigma[self.mask]
                mmodel = model.reshape((n,image.shape[0],image.shape[1]))
                mmmodel = np.empty(((len(gals) + len(srcs)+1),image[self.mask].size))
                for m in range(mmodel.shape[0]):
                    mmmodel[m] = mmodel[m][self.mask]
                op = (mmmodel/sigma[self.mask]).T
                # continue as normal with fit
                fit, chi = optimize.nnls(op,rhs)
                fits.append(fit)
            # source plane (observed) magnitudes
            if len(self.srcs)==1:
                magg = srcs[0].getMag(fits[0][-2],self.ZPs[0])
                magr = srcs[0].getMag(fits[1][-2],self.ZPs[1])
                magi = srcs[0].getMag(fits[2][-2],self.ZPs[2])
            elif self.srcno == 2.:
                if np.any(fits[0][-3:-1]==0):
                    ii = np.where(fits[0][-3:-1] !=0)[0]
                    magg = srcs[ii].getMag(fits[0][ii],self.ZPs[0])
                else:
                    mg1,mg2 = srcs[0].getMag(fits[0][-3],self.ZPs[0]), srcs[1].getMag(fits[0][-2],self.ZPs[0])
                    Fg = 10**(-0.4*mg1) + 10**(-0.4*mg2)
                    magg =  -2.5*np.log10(Fg)
                if np.any(fits[1][-3:-1]==0):
                    ii = np.where(fits[1][-3:-1] !=0)[0]
                    magr = srcs[ii].getMag(fits[1][ii],self.ZPs[1])
                else:
                    mr1,mr2 = srcs[0].getMag(fits[1][-3],self.ZPs[1]), srcs[1].getMag(fits[1][-2],self.ZPs[1])
                    Fr = 10**(-0.4*mr1) + 10**(-0.4*mr2)
                    magr =  -2.5*np.log10(Fr)
                if np.any(fits[2][-3:-1]==0):
                    ii = np.where(fits[2][-3:-1] !=0)[0]
                    magi = srcs[ii].getMag(fits[2][ii],self.ZPs[2])
                else:
                    mi1,mi2 = srcs[0].getMag(fits[2][-3],self.ZPs[2]), srcs[1].getMag(fits[2][-2],self.ZPs[2])
                    Fi = 10**(-0.4*mi1) + 10**(-0.4*mi2)
                    magi =  -2.5*np.log10(Fi)
            # source sizes
            if self.srcno == 1:
                Reg = Rei = Rer = srcs[0].pars['re']*0.263
            elif self.srcno == 2:
                Xgrid = np.logspace(-4,5,1501)
                Ygrid = np.logspace(-4,5,1501)
                bandRes = []
                for i in range(len(self.imgs)):
                    source = fits[i][-3]*srcs[0].eval(Xgrid) + fits[i][-2]*srcs[1].eval(Xgrid)
                    R = Xgrid.copy()
                    light = source*2.*np.pi*R
                    mod = splrep(R,light,t=np.logspace(-3.8,4.8,1301))
                    intlight = np.zeros(len(R))
                    for i in range(len(R)):
                        intlight[i] = splint(0,R[i],mod)
                    model = splrep(intlight[:-300],R[:-300])
                    if len(model[1][np.where(np.isnan(model[1])==True)]>0):
                        model = splrep(intlight[:-600],R[:-600])
                    reff = splev(0.5*intlight[-1],model)
                    bandRes.append(reff*0.263)
                Reg, Rer, Rei = bandRes
            # source surface brightnesses
            mug = magg +  2.5*np.log10(2.*np.pi*Reg**2.)
            mur = magr + 2.5*np.log10(2.*np.pi*Rer**2.)
            mui = magi + 2.5*np.log10(2.*np.pi*Rei**2.)
            self.muPDF.append(np.array([mug,mur,mui]))
	    if kpc:
		self.RePDF.append(np.array([Reg*self.scale,Rer*self.scale,Rei*self.scale]))
	    else:
		self.RePDF.append([Reg, Rer, Rei])
            self.magPDF.append(np.array([magg, magr, magi]))
            self.fitPDF.append(np.array(fits))
            self.grPDF.append(magg-magr)
            self.giPDF.append(magg-magi)
        np.save('/data/ljo31b/lenses/analysis/PDFs',[self.muPDF,self.RePDF,self.magPDF,self.fitPDF,self.grPDF,self.giPDF])

    def EasyAddSourcePDFs(self):
        self.muPDF, self.RePDF, self.magPDF, self.fitPDF, self.grPDF, self.giPDF = np.load('/data/ljo31b/lenses/analysis/PDFs.npy')

    def EasyAddLensPDFs(self):
        self.lensmuPDF, self.lensRePDF, self.lensmagPDF,self.fitlensPDF, self.lensgrPDF, self.lensgiPDF = np.load('/data/ljo31b/lenses/analysis/LensPDFs.npy')

    def GetLensPDFs(self,kpc=False):
        try:
            self.EasyAddSourcePDFs()
        except:
            print 'need to get source PDFs first!'
        self.lensmuPDF = []
        self.lensRePDF = []
        self.lensmagPDF = []
        self.lensfitPDF = self.fitPDF
        self.lensgrPDF = []
        self.lensgiPDF = []
        # lens mags
        for fits in self.lensfitPDF:
            if self.galno == 1:
                lensmagg = gals[0].getMag(fits[0][0],self.ZPs[0])
                lensmagr = gals[0].getMag(fits[1][0],self.ZPs[1])
                lensmagi = gals[0].getMag(fits[2][0],self.ZPs[2])
            elif self.galno == 2.:
                if np.any(fits[0][0:2]==0):
                    ii = np.where(fits[0][0:2] !=0)[0]
                    lensmagg = gals[ii].getMag(fits[0][ii],self.ZPs[0])
                else:
                    mg1,mg2 = gals[0].getMag(fits[0][0],self.ZPs[0]), gals[1].getMag(fits[0][1],self.ZPs[0])
                    Fg = 10**(-0.4*mg1) + 10**(-0.4*mg2)
                    lensmagg =  -2.5*np.log10(Fg)
                if np.any(fits[1][0:2]==0):
                    ii = np.where(fits[1][0:2] !=0)[0]
                    lensmagr = gals[ii].getMag(fits[1][ii],self.ZPs[1])
                else:
                    mr1,mr2 = gals[0].getMag(fits[1][0],self.ZPs[1]), gals[1].getMag(fits[1][1],self.ZPs[1])
                    Fr = 10**(-0.4*mr1) + 10**(-0.4*mr2)
                    lensmagr =  -2.5*np.log10(Fr)
                if np.any(fits[2][0:2]==0):
                    ii = np.where(fits[2][0:2] !=0)[0]
                    lensmagi = gals[ii].getMag(fits[2][ii],self.ZPs[2])
                else:
                    mi1,mi2 = gals[0].getMag(fits[2][0],self.ZPs[2]), gals[1].getMag(fits[2][1],self.ZPs[2])
                    Fi = 10**(-0.4*mi1) + 10**(-0.4*mi2)
                    galmagi =  -2.5*np.log10(Fi)
            # lens sizes
            if self.galno == 1:
                lensReg = lensRei = lensRer = gals[0].pars['re']*0.263
            elif self.galno == 2:
                Xgrid = np.logspace(-4,5,1501)
                Ygrid = np.logspace(-4,5,1501)
                bandRes = []
                for i in range(len(self.imgs)):
                    galaxy = fits[i][0]*gals[0].eval(Xgrid) + fits[i][1]*gals[1].eval(Xgrid)
                    R = Xgrid.copy()
                    light = galaxy*2.*np.pi*R
                    mod = splrep(R,light,t=np.logspace(-3.8,4.8,1301))
                    intlight = np.zeros(len(R))
                    for i in range(len(R)):
                        intlight[i] = splint(0,R[i],mod)
                    model = splrep(intlight[:-500],R[:-500])
                    if len(model[1][np.where(np.isnan(model[1])==True)]>0):
                        model = splrep(intlight[:-600],R[:-600])
                    reff = splev(0.5*intlight[-1],model)
                    bandRes.append(reff*0.263)
                lensReg, lensRer, lensRei = bandRes
            # lens SBs
            lensmug = lensmagg +  2.5*np.log10(2.*np.pi*lensReg**2.)
            lensmur = lensmagr + 2.5*np.log10(2.*np.pi*lensRer**2.)
            lensmui = lensmagi + 2.5*np.log10(2.*np.pi*lensRei**2.)
            # append alles
            self.lensmuPDF.append(np.array([lensmug,lensmur,lensmui]))
	    if kpc:
		self.lensRePDF.append(np.array([lensReg*self.lensscale,lensRer*self.lensscale,lensRei*self.lensscale]))
	    else:
		self.lensRePDF.append([lensReg, lensRer, lensRei])
            self.lensmagPDF.append(np.array([lensmagg, lensmagr, lensmagi]))
            self.lensgrPDF.append(lensmagg-lensmagr)
            self.lensgiPDF.append(lensmagg-lensmagi)
        np.save('/data/ljo31b/lenses/analysis/LensPDFs',[self.lensmuPDF,self.lensRePDF,self.lensmagPDF,self.lensfitPDF,self.lensgrPDF,self.lensgiPDF])


    def UncertaintiesFromSourcePDFs(self): 
        magPDF,muPDF,RePDF,grPDF,giPDF = np.array(self.magPDF),np.array(self.muPDF),np.array(self.RePDF),np.array(self.grPDF),np.array(self.giPDF)
        size = len(magPDF)
        maggPDF = [magPDF[i][0] for i in range(size)]
        magrPDF = [magPDF[i][1] for i in range(size)]
        magiPDF = [magPDF[i][2] for i in range(size)]
        mugPDF = [muPDF[i][0] for i in range(size)]
        murPDF = [muPDF[i][1] for i in range(size)]
        muiPDF = [muPDF[i][2] for i in range(size)]
        regPDF = [RePDF[i][0] for i in range(size)]
        rerPDF = [RePDF[i][1] for i in range(size)]
        reiPDF = [RePDF[i][2] for i in range(size)]
        PDFs = [maggPDF,magrPDF,magiPDF,mugPDF,murPDF,muiPDF,regPDF,rerPDF,reiPDF,grPDF,giPDF]
        meds, diffs = [], []
        for PDF in PDFs:
            L,M,U = np.percentile(PDF,16), np.percentile(PDF,50),np.percentile(PDF,84)
            V = [M-L, M, U-M]
            meds.append(V[1])
            diffs.append(np.mean((V[0],V[2])))
        self.diffs = diffs
        self.meds = meds
        return self.meds,self.diffs

                         
    def UncertaintiesFromLensPDFs(self):
        magPDF,muPDF,RePDF,grPDF,giPDF = np.array(self.lensmagPDF),np.array(self.lensmuPDF),np.array(self.lensRePDF),np.array(self.lensgrPDF),np.array(self.lensgiPDF)
        size = len(magPDF)
        maggPDF = [magPDF[i][0] for i in range(size)]
        magrPDF = [magPDF[i][1] for i in range(size)]
        magiPDF = [magPDF[i][2] for i in range(size)]
        mugPDF = [muPDF[i][0] for i in range(size)]
        murPDF = [muPDF[i][1] for i in range(size)]
        muiPDF = [muPDF[i][2] for i in range(size)]
        regPDF = [RePDF[i][0] for i in range(size)]
        rerPDF = [RePDF[i][1] for i in range(size)]
        reiPDF = [RePDF[i][2] for i in range(size)]
        PDFs = [maggPDF,magrPDF,magiPDF,mugPDF,murPDF,muiPDF,regPDF,rerPDF,reiPDF,grPDF,giPDF]
        meds, diffs = [], []
        for PDF in PDFs:
            L,M,U = np.percentile(PDF,16), np.percentile(PDF,50),np.percentile(PDF,84)
            V = [M-L, M, U-M]
            meds.append(V[1])
            diffs.append(np.mean((V[0],V[2])))
        self.lensdiffs = diffs
        self.lensmeds = meds
        return self.lensmeds,self.lensdiffs


      
result = np.load('/data/ljo31b/lenses/model_gri_gri_B_2')
model = Lens(result)
model.Initialise()
model.GetSourceMags()
model.GetSourceSize(kpc=True)
model.GetSourceSB()
model.GetLensMags()
model.GetLensSize(kpc=True)
model.GetLensSB()
model.MakePDFDict()
#model.GetPDFs(kpc=True)
model.EasyAddSourcePDFs()
meds, diffs = model.UncertaintiesFromSourcePDFs()
np.save('/data/ljo31b/lenses/analysis/meds_diffs',[meds,diffs])
model.GetLensPDFs(kpc=True)
meds, diffs = model.UncertaintiesFromLensPDFs()
np.save('/data/ljo31b/lenses/analysis/lens_meds_diffs',[meds,diffs])


'''result = np.load('/data/ljo31b/lenses/model_gri_gri_B_2')
lp,trace,dic,_= result
a2=0
a1,a3 = numpy.unravel_index(lp[:,0].argmax(),lp[:,0].shape)

'''
