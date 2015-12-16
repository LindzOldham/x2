import pymc
from imageSim import SBModels
from pylens import MassModels

class LFObject:
    def __init__(self,x,y,name):
        self.name = name
        self.pars = self.initializePars(x,y)
        self.model = None
        self.modelPars = None
        self.cov = None

    def initializePars(self,x0,y0):
        return

    def reset(self):
        self.model = None
        self.modelPars = None
        self.cov = None

    def makeModel(self):
        inPars = {}
        self.modelPars = []
        self.cov = []
        for key in self.keys:
            p,c = self.getPar(key)
            inPars[key] = p
            if c is not None:
                self.modelPars.append(p)
                self.cov.append(c)
        self.model = self.modelClass(self.name,inPars)

    def getPar(self,key):
        name = '%s:%s'%(key,self.name)
        p = self.pars[key]
        if p['type']=='uniform':
            v = pymc.Uniform(name,p['lower'],p['upper'],p['value'])
            c = p['sdev']
        elif p['type']=='normal':
            if key=='q':
                v = pymc.TruncatedNormal(name,p['mean'],p['sigma']**-2,0.,1.,p['value'])
            elif key=='n':
                v = pymc.TruncatedNormal(name,p['mean'],p['sigma']**-2,0.,10.,p['value'])
            elif key=='eta':
                v = pymc.TruncatedNormal(name,p['mean'],p['sigma']**-2,0.,2.,p['value'])
            v = pymc.Normal(name,p['mean'],p['sigma']**-2,p['value'])
            c = p['sdev']
        else:
            v = p['value']
            c = None
        return v,c

    def delete(self):
        pass


class Galaxy(LFObject):
    def __init__(self,x,y,name,manager):
        self.keys = ['x','y','q','pa','re','n']
        self.modelClass = SBModels.Sersic
        self.manager = manager
        LFObject.__init__(self,x,y,name)

    def initializePars(self,x0,y0):
        p = {}
        p['x'] = {'type':'constant','value':x0,'lower':x0-5.,'upper':x0+5.,
                'mean':x0,'sigma':5.,'sdev':0.1}
        p['y'] = {'type':'constant','value':y0,'lower':y0-5.,'upper':y0+5.,
                'mean':y0,'sigma':5.,'sdev':0.1}
        p['q'] = {'type':'constant','value':0.9,'lower':0.05,'upper':1.,
                'mean':0.8,'sigma':0.1,'sdev':0.05}
        p['pa'] = {'type':'constant','value':0.,'lower':-180.,'upper':180.,
                'mean':0.,'sigma':10.,'sdev':1.}
        p['re'] = {'type':'constant','value':5.,'lower':0.5,'upper':100.,
                'mean':10.,'sigma':1.,'sdev':0.5}
        p['n'] = {'type':'constant','value':2.5,'lower':0.5,'upper':8.,
                'mean':2.5,'sigma':0.5,'sdev':0.1}
        return p


class Lens(LFObject):
    def __init__(self,x,y,name,manager):
        self.keys = ['x','y','q','pa','b','eta']
        self.modelClass = MassModels.PowerLaw
        self.manager = manager
        LFObject.__init__(self,x,y,name)

    def initializePars(self,x0,y0):
        p = {}
        p['x'] = {'type':'constant','value':x0,'lower':x0-5.,'upper':x0+5.,
                'mean':x0,'sigma':5.,'sdev':0.1}
        p['y'] = {'type':'constant','value':y0,'lower':y0-5.,'upper':y0+5.,
                'mean':y0,'sigma':5.,'sdev':0.1}
        p['q'] = {'type':'constant','value':0.9,'lower':0.05,'upper':1.,
                'mean':0.8,'sigma':0.1,'sdev':0.05}
        p['pa'] = {'type':'constant','value':0.,'lower':-180.,'upper':180.,
                'mean':0.,'sigma':10.,'sdev':1.}
        p['b'] = {'type':'constant','value':15.,'lower':1.,'upper':100.,
                'mean':15.,'sigma':1.,'sdev':0.1}
        p['eta'] = {'type':'constant','value':1.,'lower':0.5,'upper':1.5,
                'mean':1.,'sigma':0.1,'sdev':0.05}
        return p


class Shear(LFObject):
    def __init__(self,x,y):
        self.keys = ['x','y','b','pa']
        self.modelClass = MassModels.ExtShear
        LFObject.__init__(self,x,y,'External shear')

    def initializePars(self,x0,y0):
        p = {}
        p['x'] = {'type':'constant','value':x0,'lower':x0-5.,'upper':x0+5.,
                'mean':x0,'sigma':5.,'sdev':0.1}
        p['y'] = {'type':'constant','value':y0,'lower':y0-5.,'upper':y0+5.,
                'mean':y0,'sigma':5.,'sdev':0.1}
        p['b'] = {'type':'constant','value':0.0,'lower':-0.2,'upper':0.2,
                'mean':0.,'sigma':0.05,'sdev':0.01}
        p['pa'] = {'type':'constant','value':0.,'lower':-180.,'upper':180.,
                'mean':0.,'sigma':10.,'sdev':1.}
        return p

    def getSave(self):
        return self.pars

    def load(self,pars):
        self.pars = pars
