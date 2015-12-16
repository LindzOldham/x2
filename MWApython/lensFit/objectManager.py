import pymc
import numpy
import Tkinter as Tk
from imageSim import SBModels
from pylens import MassModels
from lfParTables import ParTable
from lfObjects import *

class ObjectManager:
    def __init__(self,parent):
        self.parent = parent
        self.nobj = 0
        self.objs = {}
        self.table = ParTable(self)
        self.tableVisible = False

    def deleteObject(self,name):
        del self.objs[name]
        self.nobj -= 1
        # Rename objects so they aren't overwritten
        keys = self.objs.keys()
        keys.sort()
        objs = {}
        for i in range(len(keys)):
            newkey = '%s %d'%(self.label,i+1)
            objs[newkey] = self.objs[keys[i]]
            objs[newkey].name = newkey
        self.objs = objs
        if self.tableVisible:
            if len(self.objs)==0:
                self.table.closeTable()
            else:
                self.table.drawTable()
        self.parent.fitLight()

    def tableClosed(self):
        self.tableVisible = False
        return

    def load(self,objs):
        self.nobj = len(objs)
        self.objs = {}
        keys = objs.keys()
        keys.sort()
        for name in keys:
            pars = objs[name]
            self.objs[name] = self.lfObject(0,0,name,self)
            self.objs[name].pars = pars


    def getSave(self):
        objs = {}
        for name,obj in self.objs.iteritems():
            obj.reset()
            objs[name] = obj.pars
        return objs

    def addModel(self,x,y):
        self.nobj += 1
        name = "%s %d"%(self.label,self.nobj)
        self.objs[name] = self.lfObject(x,y,name,self)
        self.parent.fitLight()
        return name


class GalaxyManager(ObjectManager):
    def __init__(self,parent):
        ObjectManager.__init__(self,parent)
        self.label = 'Galaxy'
        self.pars = ['x','y','q','pa','re','n']
        self.lfObject = Galaxy

    def addGalaxy(self,x,y):
        print self.nobj
        self.nobj += 1
        name = 'Galaxy %d'%(self.nobj)
        self.objs[name] = Galaxy(x,y,name,self)
#        self.gui.glist.insert(
#        self.parent.gui.showGalaxyTable(self.objs[name])
#        self.table.drawTable(self.parent.gui.ginfoF)
        self.parent.fitLight()
        return name


class LensManager(ObjectManager):
    def __init__(self,parent):
        ObjectManager.__init__(self,parent)
        self.label = 'Lens'
        self.pars = ['x','y','q','pa','b','eta']
        self.lfObject = Lens

    def addLens(self,x,y):
        self.nobj += 1
        name = 'Lens %d'%(self.nobj)
        self.objs[name] = Lens(x,y,name,self)
        self.table.drawTable()
        #self.parent.fitLight()

class SrcManager(ObjectManager):
    def __init__(self,parent):
        ObjectManager.__init__(self,parent)
        self.label = 'Source'
        self.pars = ['x','y','q','pa','re','n']
        self.lfObject = Galaxy

    def addSrc(self,x,y):
        self.nobj += 1
        name = 'Source %d'%(self.nobj)
        self.objs[name] = Galaxy(x,y,name,self)
        self.table.drawTable()
        self.parent.fitLight()

