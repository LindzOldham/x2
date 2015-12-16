from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import Tkinter as Tk
import ttk
import Tix
from Tkconstants import *
import traceback, tkMessageBox
import lfImageWindow
import lfParTables

TCL_DONT_WAIT           = 1<<1
TCL_WINDOW_EVENTS       = 1<<2
TCL_FILE_EVENTS         = 1<<3
TCL_TIMER_EVENTS        = 1<<4
TCL_IDLE_EVENTS         = 1<<5
TCL_ALL_EVENTS          = 0


class ObjectTab:
    def __init__(self,pane,gui):
        self.toppane = pane
        self.gui = gui

        listf = ttk.Labelframe(pane,text='Models',width=300)
        infof = ttk.Labelframe(pane,text='Info',width=340)

        listf.pack_propagate(0)
        infof.pack_propagate(0)

        listf.pack(side=LEFT,expand=1,fill=BOTH)
        infof.pack(side=RIGHT,expand=1,fill=BOTH)

        list = Tk.Listbox(listf,listvariable=Tk.StringVar(value=()))
        list.pack(side=TOP)
        list.bind('<<ListboxSelect>>',self.selectModel)
        addButton = Tk.Button(listf,text='Add New %s'%self.objtype,command=self.addModel)
        addButton.pack(side=BOTTOM)

        infoF = Tk.Frame(infof,width=420)
        infoF.pack(side=BOTTOM,expand=1)
        self.addButton = addButton
        self.listf = listf
        self.list = list
        self.listn = 0
        self.infof = infof
        self.infoF = infoF
        self.infoPane = None


    def addModel(self):
        if self.gui.activeButton==self.addButton:
            self.gui.deactivateButtons()
            return
        self.gui.deactivateButtons()
        if self.gui.imgWin.visible==False:
            self.gui.imgWin.drawWindow()

        def onPress(event):
            axes = event.inaxes
            if axes==self.gui.imgWin.a1 or axes==self.gui.imgWin.a2:
                name = self.objectManager.addModel(event.xdata,event.ydata)
                self._addModel(name)
                self.gui.deactivateButtons()
        self.gui.imgWin.pid = self.gui.imgWin.canvas.mpl_connect('button_press_event',
                                                            onPress)
        self.addButton.configure(text='Cancel')
        self.gui.activeButton = self.addButton


    def _addModel(self,name):
        self.list.insert(self.listn,name)
        self.list.activate(self.listn)
        self.list.selection_clear(0,END)
        self.list.selection_set(self.listn)
        self.showTable(name)
        self.listn += 1


    def selectModel(self,event):
        name = self.list.get(self.list.curselection())
        self.list.activate(self.list.curselection())
        self.showTable(name)


    def showTable(self,name=None):
        import lfParTables
        if self.infoPane is not None:
            self.infoPane.destroy()
        object = self.objectManager.objs[name]
        self.infoPane = lfParTables.TableEntry(self.infoF,object,self)


    def removeModel(self):
        if self.listn==0:
            return None
        removedID = self.list.index(ACTIVE)
        self.list.delete(removedID)
        self.list.delete(0,END)
        self.listn -= 1
        for i in range(self.listn):
            self.list.insert(i,'%s %s'%(self.objtype,i+1))
        self.list.selection_clear(0,END)
        if self.listn>removedID:
            self.list.activate(removedID)
            self.list.selection_set(removedID)
            self.showTable(self.list.get(removedID))
        elif self.listn>0:
            self.list.activate(END)
            self.list.selection_set(END)
            self.showTable(self.list.get(END))
        return 0

    def resetAddButton(self):
        self.addButton.configure(text='Add %s'%(self.objtype))






class GalaxyTab(ObjectTab):
    def __init__(self,pane,gui):
        self.objectManager = gui.parent.galaxyManager
        self.objtype = 'Galaxy'
        ObjectTab.__init__(self,pane,gui)


class LensTab(ObjectTab):
    def __init__(self,pane,gui):
        self.objectManager = gui.parent.lensManager
        self.objtype = 'Lens'
        ObjectTab.__init__(self,pane,gui)

        self.shearFlag = Tk.IntVar()
        self.shearFlag.set(0)
        shearCheck = Tk.Checkbutton(self.listf,text='Use shear?',onvalue=1,
                        offvalue=0,variable=self.shearFlag,
                        command=self.setShear)
        shearCheck.pack(side=BOTTOM)
        shearFrame = Tk.Frame(self.listf,width=self.listf.winfo_width())
        shearFrame.pack(side=BOTTOM)
        self.shearFrame = shearFrame
        shear = gui.parent.shear
        self.shearRow1 = lfParTables.TableRow(shearFrame,'Shear',0,shear.pars['b'],gui.parent)
        self.shearRow2 = lfParTables.TableRow(shearFrame,'Angle',1,shear.pars['pa'],gui.parent)
        for child in self.shearFrame.winfo_children():
            child.configure(state=DISABLED)

    def setShear(self,refit=True):
        if self.shearFlag.get()==1:
            for child in self.shearFrame.winfo_children():
                child.configure(state=NORMAL)
            self.shearRow1.changeType(doCheck=False)
            self.shearRow2.changeType(doCheck=False)
            self.gui.parent.shearFlag = True
        else:
            for child in self.shearFrame.winfo_children():
                child.configure(state=DISABLED)
            self.gui.parent.shearFlag = False
        if refit:
            self.gui.parent.fitLight()

    def updateShear(self):
        shear = self.gui.parent.shear
        self.shearRow1 = lfParTables.TableRow(self.shearFrame,'Shear',0,shear.pars['b'],self.gui.parent)
        self.shearRow2 = lfParTables.TableRow(self.shearFrame,'Angle',1,shear.pars['pa'],self.gui.parent)



class SourceTab(ObjectTab):
    def __init__(self,pane,gui):
        self.objectManager = gui.parent.srcManager
        self.objtype = 'Source'
        ObjectTab.__init__(self,pane,gui)

