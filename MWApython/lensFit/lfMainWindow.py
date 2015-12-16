from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import Tkinter as Tk
import ttk
import Tix
from Tkconstants import *
import traceback, tkMessageBox
import lfImageWindow
import lfObjectTab

TCL_DONT_WAIT           = 1<<1
TCL_WINDOW_EVENTS       = 1<<2
TCL_FILE_EVENTS         = 1<<3
TCL_TIMER_EVENTS        = 1<<4
TCL_IDLE_EVENTS         = 1<<5
TCL_ALL_EVENTS          = 0


class LensGUI:
    def __init__(self,parent,top):
        self.parent = parent
        self.root = top
        self.imgWin = None
        self.exit = -1
        self.activeButton = None
        self.build()
        #self.loop()
        #self.destroy()


    def MainMenu(self):
        top = self.root
        win = Tk.Frame(top,bd=2,relief=RAISED)

        file = ttk.Menubutton(win,text='File',underline=0)
        file.pack(side=LEFT)
        filemenu = Tk.Menu(file,tearoff=0)
        file['menu'] = filemenu

        filemenu.add_command(label='Save',command=self.parent.saveState,underline=0)
        filemenu.add_command(label='Load',command=self.parent.loadState,underline=0)
        filemenu.add_separator()
        filemenu.add_command(label='Exit',command=self.quitcmd,underline=0)


        fitLightButton = Tix.Button(win,text='Fit Light',command=self.parent.fitLight)
        fitLightButton.pack(side=LEFT)
        optimizeButton = Tix.Button(win,text='Optimize',command=self.parent.optimize)
        optimizeButton.pack(side=LEFT)
        #longoptButton = Tix.Button(win,text='Long optimize',command=self.parent.loptimize)
        #longoptButton.pack(side=LEFT)

        showimgButton = Tix.Button(win,text='Show Image',command=self.showImg,relief=FLAT)
        showimgButton.pack(side=RIGHT)

        return win

    def MainPanel(self):
        top = self.root

        win = ttk.Notebook(top,name='win')

        #infotab = Tk.Frame(win)
        galtab = Tk.Frame(win)
        lenstab = Tk.Frame(win)
        srctab = Tk.Frame(win)
        #win.add(infotab,text='Info')
        win.add(galtab,text='Galaxies')
        win.add(lenstab,text='Lenses')
        win.add(srctab,text='Sources')

        win.pack(expand=1, fill=Tix.BOTH, padx=5, pady=5 ,side=TOP)
        return win


    def showImg(self):
        if self.imgWin is None:
            self.imgWin = lfImageWindow.ImageWindow(self.parent)
        elif self.imgWin.visible==False:
            self.imgWin.draw_window()
        else:
            self.imgWin.update_window()


    def build(self):
        root = self.root
        z = root.winfo_toplevel()
        z.wm_title('Lens Modeller')
        z.geometry('640x480+10+10')
        menu = self.MainMenu()
        panel = self.MainPanel()
        menu.pack(side=TOP,fill=X)
        panel.pack(side=BOTTOM,fill=BOTH,expand=1)

        galTab = lfObjectTab.GalaxyTab(panel.children[panel.tabs()[0][5:]],self)
        lensTab = lfObjectTab.LensTab(panel.children[panel.tabs()[1][5:]],self)
        srcTab = lfObjectTab.SourceTab(panel.children[panel.tabs()[2][5:]],self)

        self.galTab = galTab
        self.lensTab = lensTab
        self.srcTab = srcTab
        self.objTabs = [galTab,lensTab,srcTab]

        self.showImg()

        z.wm_protocol('WM_DELETE_WINDOW',lambda self=self: self.quitcmd())


    def quitcmd(self):
        self.exit = 0

    def loop(self):
        import sys
        while self.exit<0:
            try:
                while self.exit<0:
                    self.root.tk.dooneevent(TCL_ALL_EVENTS)
            except SystemExit:
                self.exit = 1
                return
            except KeyboardInterrupt:
                if tkMessageBox.askquestion('Interrupt','Quit?')=='yes':
                    self.exit = 1
                    return
                continue
            except:
                t, v, tb = sys.exc_info()
                text = ""
                for line in traceback.format_exception(t,v,tb):
                    text += line + '\n'
                try:
                    tkMessageBox.showerror ('Error', text)
                except:
                    pass
                self.exit = 1
                raise SystemExit, 1


    def destroy(self):
        self.root.destroy()

    def rebuild(self):
        for tab in self.objTabs:
            if tab.infoPane is not None:
                tab.infoPane.destroy()
                tab.infoPane = None
            while tab.removeModel() is not None:
                pass

        for manager,tab in zip(self.parent.managers,self.objTabs):
            for key in sorted(manager.objs.keys()):
                tab._addModel(key)

        self.lensTab.updateShear()
        self.lensTab.shearFlag.set(self.parent.shearFlag)
        self.lensTab.setShear()


    def deactivateButtons(self):
#        if self.activeButton==None:
#            return
        if self.imgWin is not None and self.imgWin.toolbar.mode!='':
            self.imgWin.toolbar.zoom()
            self.imgWin.toolbar.pan()
            self.imgWin.toolbar.pan()
        for tab in self.objTabs:
            if self.activeButton==tab.addButton:
                tab.resetAddButton()
                self.imgWin.canvas.mpl_disconnect(self.imgWin.pid)
        self.imgWin.pid = None
        self.activeButton = None

