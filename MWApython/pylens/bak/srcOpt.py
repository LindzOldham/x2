import pymc,sys,cPickle
import numpy
from math import pi,log10
import pyfits
from scipy import ndimage,optimize,interpolate
from pylens import *
from imageSim import profiles,convolve,models
import time
import indexTricks as iT
import pylab
import pylens

class mover:
    def __init__(self,image,sigma,lenses,src,xc,yc,PSFfft):
        self.PSFfft = PSFfft
        self.lenses = lenses
        self.selected = 0
        self.src = src
        mod = pylens.lens_images(lenses,[src],[xc,yc],1.)
        mod = convolve.convolve(mod,self.PSFfft,False)[0]
        self.coords = [xc,yc]
        self.image = image
        self.sigma = sigma

        image[image<0.] = 0.
        image = image**0.5
        self.rawimage = pylab.imshow(image,origin='lower',interpolation='nearest')
        self.modimg = pylab.imshow(mod,origin='lower',alpha=0.6,cmap=pylab.cm.gray)
        pylab.xlim((0,image.shape[1]))
        pylab.ylim((0,image.shape[0]))
        self.canvas = self.modimg.get_figure().canvas

#        x,y = lenses[0].caustic()
#        self.caustic = pylab.plot(x,y,'k')[0]

        self.point = pylab.plot(self.src.x,self.src.y,'ko')[0]
        self.moving = False
        self.initial = None

#        self.visible = [True for i in srcs]
#        self.models = [None for i in srcs]

        self.keyid = self.canvas.mpl_connect('key_press_event',self.key_press)
        self.connect()
        pylab.show()


    def connect(self):
        """ Connect the mouse to the plot """
        self.pressid = self.canvas.mpl_connect('button_press_event',
                                                self.on_press)
        self.moveid = self.canvas.mpl_connect('motion_notify_event',
                                                self.on_motion)
        self.offid = self.canvas.mpl_connect('button_release_event',
                                                self.on_release)

    def on_press(self,event):
        """
        Deal with mouse button presses, including stretching, shifting,
            and line identification.
        """
        """ Turn off plot tools """
        if self.canvas.toolbar.mode!='':
            if event.button==2:
                self.canvas.toolbar.zoom()
                self.canvas.toolbar.pan()
                self.canvas.toolbar.pan()
            return

        self.initial = [event.xdata,event.ydata]
        self.moving = True

    def on_motion(self, event):
        """ Controls the sliding/stretching of the spectra """

        """
        Ignore this if we aren't in slide/stretch mode (ie pressing the
            mouse button
        """
        if self.moving is False:
            return

        if event.xdata is None or event.ydata is None:
            return

        if event.button==1:
            x,y = event.xdata,event.ydata
            self.point.set_xdata(x)
            self.point.set_ydata(y)
            self.src.x = x
            self.src.y = y
            mod = pylens.lens_images(self.lenses,[self.src],self.coords,1.)
            mod = convolve.convolve(mod,self.PSFfft,False)[0]
            self.point.set_data(self.src.x,self.src.y)
            self.modimg.set_data(mod)
            self.modimg.autoscale()
            pylab.draw()
        elif event.button==2:
            from math import atan2,pi
            x0,y0 = self.initial
            dy = event.ydata-y0
            dx = event.xdata-x0
            angle = (atan2(dy,dx)*180./pi)%180.
            self.src.pa = angle
            mod = pylens.lens_images(self.lenses,[self.src],self.coords,1.)
            mod = convolve.convolve(mod,self.PSFfft,False)[0]
            self.modimg.set_data(mod)
            pylab.draw()
        elif event.button==3:
            x,y = event.xdata,event.ydata
            x0,x1 = pylab.xlim()
            y0,y1 = pylab.ylim()
            dx = x1-x0
            q = (x-x0)/dx
            dy = y1-y0
            yi = (self.initial[1]-y0)
            f = (y-y0)/yi
            self.src.q = q
            self.src.scale = f
            mod = pylens.lens_images(self.lenses,[self.src],self.coords,1.)
            mod = convolve.convolve(mod,self.PSFfft,False)[0]
            self.modimg.set_data(mod)
            pylab.draw()


    def on_release(self,event):
        if self.moving==True:
            self.moving = False

    def key_press(self,event):
        if event.key.lower()=='b':
            txt = raw_input('Enter new lens strength (%.1f):'%self.lenses[0].b)
            try:
                b = float(txt)
            except:
                print 'Invalid option: %s'%txt
                return
            self.lenses[0].b = b
            mod = pylens.lens_images(self.lenses,[self.src],self.coords,1.)
            mod = convolve.convolve(mod,self.PSFfft,False)[0]
            x,y = self.lenses[0].caustic()
            self.caustic.set_data(x,y)
            self.modimg.set_data(mod)
        if event.key.lower()=='p':
            self.canvas.toolbar.pan()
            print self.src.x,self.src.y
            return
            self.canvas.toolbar.pan()
            txt = raw_input('New PA (%.2f):'%self.lenses[0].pa)
            try:
                pa = float(txt)
            except:
                print 'Invalid option: %s'%txt
                return
            self.lenses[0].pa = pa
            mod = pylens.lens_images(self.lenses,[self.src],self.coords,1.)
            mod = convolve.convolve(mod,self.PSFfft,False)[0]
            x,y = self.lenses[0].caustic()
            self.caustic.set_data(x,y)
            self.modimg.set_data(mod)
        if event.key.lower()=='q':
            txt = raw_input('New axis ratio (%.2f):'%self.lenses[0].q)
            try:
                q = float(txt)
            except:
                print 'Invalid option: %s'%txt
                return
            self.lenses[0].q = q
            mod = pylens.lens_images(self.lenses,[self.src],self.coords,1.)
            mod = convolve.convolve(mod,self.PSFfft,False)[0]
            x,y = self.lenses[0].caustic()
            self.caustic.set_data(x,y)
            self.modimg.set_data(mod)
        if event.key.lower()=='c':
            txt = raw_input('Enter source (1-%d):'%len(self.srcs))
            try:
                i = int(txt)-1
            except:
                print 'Invalid option: %s'%txt
                return
            self.src = self.srcs[i]
            self.selected = i
            mod = pylens.lens_images(self.lenses,[self.src],self.coords,1.)
            mod = convolve.convolve(mod,self.PSFfft,False)[0]
            self.point.set_data(self.src.x,self.src.y)
            self.modimg.set_data(mod)
        if event.key.lower()=='d':
            if self.visible[self.selected]==False:
                print "Component Added"
                self.image += self.models[self.selected]
                self.rawimage.set_data(self.image)
                self.visible[self.selected] = True
            else:
                print "Component Subtracted"
                rhs = (self.image/self.sigma).flatten()
                mod = self.modimg.get_array().data
                op = numpy.atleast_2d((mod/self.sigma).flatten()).T
                fit = numpy.linalg.lstsq(op,rhs)[0]
                self.models[self.selected] = mod*fit
                self.image -= mod*fit
                self.rawimage.set_data(self.image)
                self.visible[self.selected] = False
        if event.key.lower()=='o':
            self.canvas.toolbar.zoom()
            lens = self.lenses[0]
            shear = self.lenses[1]
            src = self.src
            SX = pymc.Uniform('SX',src.x-10.,src.x+10.,value=src.x)
            SY = pymc.Uniform('SY',src.y-10.,src.y+10.,value=src.y)
            SQ = pymc.Uniform('SQ',0.1,1.,value=src.q)
            SP = pymc.Uniform('SP',-180.,180.,value=src.pa)
            try:
                SR = pymc.Uniform('SR',0.1,5.0,value=src.sigma)
            except:
                SR = pymc.Uniform('SR',0.1,5.0,value=src.re)


            LB = pymc.Uniform('LB',lens.b-3,lens.b+3.,value=lens.b)
            LQ = pymc.Uniform('LQ',0.2,1.,value=lens.q)
            LP = pymc.Uniform('LP',-180.,180.,value=lens.pa)

            XB = pymc.Uniform('XB',0.,0.5,value=shear.b)
            XP = pymc.Uniform('XP',-180.,180.,value=shear.pa)

            pars = [SX,SY,SQ,SP,SR,LB,LQ,LP,XB,XP]
            var = {'x':0,'y':1,'q':2,'pa':3,'re':4}
            const = {'amp':1.,'n':1.}
            tsrc = models.Sersic('source',var,const)
            const = {'x':lens.x,'y':lens.y}
            var = {'b':5,'q':6,'pa':7}
            tlens = massmodel.SIE('lens',var,const)
            var = {'b':8,'pa':9}
            tshear = massmodel.ExtShear('shear',var,const)
            cov = [0.05,0.05,0.03,0.5,0.3]
            cov += [0.05,0.03,0.5]
            cov += [0.01,0.5]
            cov = numpy.array(cov)

            xc,yc = self.coords
            @pymc.deterministic
            def logP(value=0.,p=pars):
                inpars = [p.value for p in pars]
                return lensModel.lensFit(inpars,self.image,self.sigma,[],[tlens,tshear],[tsrc],xc,yc,1.,verbose=True,psf=PSF)

            @pymc.observed
            def likelihood(value=0.,lp=logP):
                return lp

            from SampleOpt import AMAOpt as Sample6
            S = Sample6(pars,[likelihood],[logP],cov=numpy.array(cov)/10.)
            S.set_minprop(len(pars)*2)
            S.sample(2*len(pars)**2)

            logp,trace,det = S.result()
            p = trace[-1]

            d = lensModel.lensFit(p,self.image,self.sigma,[],[tlens,tshear],[tsrc],xc,yc,1.,verbose=True,psf=PSF,noResid=True)
#            pylab.figure()
#            pylab.imshow(self.image,origin='lower')
#            pylab.imshow(d,origin='lower',cmap=pylab.cm.gray,alpha=0.6)
            self.src.x = p[0]
            self.src.y = p[1]
            self.src.q = p[2]
            self.src.pa = p[3]
            try:
                self.src.sigma = p[4]
            except:
                self.src.re = p[4]
            self.lenses[0].b = p[5]
            self.lenses[0].q = p[6]
            self.lenses[0].pa = p[7]
            self.lenses[1].q = p[8]
            self.lenses[1].pa = p[9]

            x,y = self.lenses[0].caustic()
            self.caustic.set_data(x,y)
            self.point.set_data(src.x,src.y)
            mod = pylens.lens_images(self.lenses,[self.src],self.coords,1.)
            mod = convolve.convolve(mod,self.PSFfft,False)[0]
            self.modimg.set_data(mod)

        pylab.draw()


