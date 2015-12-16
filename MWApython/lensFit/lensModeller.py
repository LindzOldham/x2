import Tkinter as Tk
import ttk
import sys
import objectManager
import objectMover
import lensInference
import lfMainWindow
import lfObjects

class FitLens:
    def __init__(self,b,g=None,r=None):
        self.imgs = [b[0]]
        self.sigs = [b[1]]
        self.psfs = [b[2]]
        self.b = b[0]
        self.color = None
        if g is not None:
            import colorImage
            self.color = colorImage.ColorImage()
            self.imgs.append(g[0])
            self.sigs.append(g[1])
            self.psfs.append(g[2])
            self.g = g[0]
            if r is not None:
                self.imgs.append(r[0])
                self.sigs.append(r[1])
                self.psfs.append(r[2])
                self.r = r[0]
            else:
                self.r = self.g.copy()
                self.g = (self.r+self.b)/2.
        self.nimgs = len(self.imgs)
        self.mask = None
        self.makeColor()
        self.imgScaling = 1.
        self.resid = self.img
        self.model = self.img*0
        self.models = None


        self.root = Tk.Tk()

        self.galaxyManager = objectManager.GalaxyManager(self)
        self.lensManager = objectManager.LensManager(self)
        self.srcManager = objectManager.SrcManager(self)
        self.managers = [self.galaxyManager,self.lensManager,self.srcManager]
        self.Inference = lensInference.LensInference(self)
        self.shear = lfObjects.Shear(self.Inference.xc[0].mean(),self.Inference.yc[0].mean())
        self.shearFlag = False

        self.gui = lfMainWindow.LensGUI(self,self.root)

        self.gui.loop()
        self.gui.destroy()

        #self.root.mainloop()


    def makeColor(self):
        if self.color is None:
            self.img = self.b
        else:
            self.img = self.color.createModel(self.b,self.g,self.r)


    def saveState(self):
        import tkFileDialog
        filename = tkFileDialog.asksaveasfilename()
        if filename:
            self.save(filename)

    def save(self,filename):
        import cPickle
        o = [i.getSave() for i in [self.galaxyManager,self.lensManager,self.srcManager]]
        o.append([[p.__name__,p.parents['lower'],p.parents['upper'],p.value] for p in self.Inference.offsets])
        o.append((self.shear.getSave(),self.shearFlag))
        f = open(filename,'wb')
        cPickle.dump(o,f,2)
        f.close()

    def loadState(self):
        import tkFileDialog
        filename = tkFileDialog.askopenfilename()
        if filename:
            import cPickle
            f = open(filename)
            try:
                tmp = cPickle.load(f)
            except:
                print "Could not open file %s"%filename
            f.close()
            gals,lenses,srcs,offs,shear = tmp
            self.Inference.setOffsets(offs)
            self.galaxyManager.load(gals)
            self.lensManager.load(lenses)
            self.srcManager.load(srcs)
            self.shear.load(shear[0])
            self.shearFlag = shear[1]
            self.gui.rebuild()


    def fitLight(self):
        self.models = self.Inference.getModel()
        self.makeResid()
        self.gui.showImg()
        #self.updateDisplay(symbols=symbols)


    def optimize(self):
        li = self.Inference
        result = li.runInference()
        if result is None:
            return
        if len(li.outPars)==0:
            self.models = result
            self.makeResid()
            self.gui.showImg()
            return
        for p in li.outPars:
            if p.__name__.find(':')<0:
                continue
            par,obj = p.__name__.split(':')
            if obj[0]=='G':
                self.galaxyManager.objs[obj].pars[par]['value'] = p.value
            if obj[0]=='L':
                self.lensManager.objs[obj].pars[par]['value'] = p.value
            if obj[0]=='S':
                self.srcManager.objs[obj].pars[par]['value'] = p.value
            if obj[0]=='E':
                self.shear.pars[par]['value'] = p.value
        if self.galaxyManager.tableVisible:
            self.galaxyManager.table.drawTable()
        if self.lensManager.tableVisible:
            self.lensManager.table.drawTable()
        if self.srcManager.tableVisible:
            self.srcManager.table.drawTable()
        for tab in self.gui.objTabs:
            if tab.infoPane is not None:
                tab.infoPane.update()
        if self.shearFlag==True:
            self.gui.lensTab.updateShear()
        self.models = result
        self.makeResid()
        self.gui.showImg()


    def makeResid(self):
        if self.models is None:
            self.models = self.Inference.getModel()
        if self.models is None:
            self.resid = self.img
            self.model = self.img*0.
            return
        imgs = self.imgs
        models = self.models
        if self.color is not None:
            if self.nimgs==2:
                b = imgs[0]-models[0]
                r = imgs[1]-models[1]
                g = (b+r)/2.
                self.resid = self.color.colorize(b,g,r)
                b = models[0]
                r = models[1]
                g = (b+r)/2.
                self.model = self.color.colorize(b,g,r,newI=True)
            else:
                b = imgs[0]-models[0]
                g = imgs[1]-models[1]
                r = imgs[2]-models[2]
                self.resid = self.color.colorize(b,g,r)
                b = models[0]
                g = models[1]
                r = models[2]
                self.model = self.color.colorize(b,g,r,newI=True)
        else:
            self.resid = imgs[0]-models[0]
            self.model = models[0]


