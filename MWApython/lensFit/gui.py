from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import Tkinter as Tk

class LensGUI:
    def __init__(self,parent):
        self.root = Tk.Tk()

        self.parent = parent
        self.img = self.parent.img
        self.color = self.parent.color

        self.mover = None

        f1 = Figure((12.06,12.06))
        a1 = f1.add_axes([0,101./201,100./201,100./201])
        self.img1 = a1.imshow(self.img,origin='bottom',interpolation='nearest')
        a1.set_xticks([])
        a1.set_yticks([])
        xlim = a1.get_xlim()
        ylim = a1.get_ylim()

        a2 = f1.add_axes([101./201,101./201,100./201,100./201])
        self.img2 = a2.imshow(self.img,origin='bottom',interpolation='nearest')
        a2.set_xlim(xlim)
        a2.set_ylim(ylim)
        a2.set_xticks([])
        a2.set_yticks([])

        a3 = f1.add_axes([0.,0.,100./201,100./201])
        self.img3 = a3.imshow(self.img*0+1,origin='bottom',interpolation='nearest')
        a3.set_xlim(xlim)
        a3.set_ylim(ylim)
        a3.set_xticks([])
        a3.set_yticks([])

        a4 = f1.add_axes([101./201,0.,100./201,100./201])
        a4.imshow(self.parent.b*0)
        a4.cla()
        a4.set_xlim(xlim)
        a4.set_ylim(ylim)
        a4.set_xticks([])
        a4.set_yticks([])

        canvas = FigureCanvasTkAgg(f1,master=self.root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas,self.root )
        toolbar.update()
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        bFrame = Tk.Frame(self.root)
        bFrame.pack(side=Tk.TOP,fill=Tk.BOTH,expand=1)

        self.f1 = f1
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.bFrame = bFrame

        self.canvas = canvas
        self.toolbar = toolbar

        self.rubberBox = None

        self.addButtons()

    def addButtons(self):
        self.activeButton = None
        self.bAGtext = Tk.StringVar()
        self.bAGtext.set('Add Galaxy')
        self.buttonAG = Tk.Button(self.toolbar,textvariable=self.bAGtext,command=self.parent.addGal,width=10)
        self.buttonAG.pack(side=Tk.LEFT)

        self.bALtext = Tk.StringVar()
        self.bALtext.set('Add Lens')
        self.buttonAL = Tk.Button(self.toolbar,textvariable=self.bALtext,command=self.parent.addLens,width=10)
        self.buttonAL.pack(side=Tk.LEFT)

        self.bAStext = Tk.StringVar()
        self.bAStext.set('Add Source')
        self.buttonAS = Tk.Button(self.toolbar,textvariable=self.bAStext,command=self.parent.addSrc,width=10)
        self.buttonAS.pack(side=Tk.LEFT)
        self.buttonAS.configure(state='disabled')

        self.buttonFit = Tk.Button(self.toolbar,text='Fit Light',command=self.parent.fitLight,width=10)
        self.buttonFit.pack(side=Tk.LEFT)
        #self.buttonFit.configure(state='disabled')

        self.bOpttext = Tk.StringVar()
        self.bOpttext.set('Optimize')
        self.buttonOptimize = Tk.Button(self.toolbar,textvariable=self.bOpttext,command=self.parent.optimize,width=10)
        self.buttonOptimize.pack(side=Tk.LEFT)
        #self.buttonOptimize.configure(state='disabled')

        self.buttonSave = Tk.Button(self.bFrame,text='Save',command=self.parent.saveState,width=10)
        self.buttonSave.pack(side=Tk.LEFT)

        self.buttonLoad = Tk.Button(self.bFrame,text='Load',command=self.parent.loadState,width=10)
        self.buttonLoad.pack(side=Tk.LEFT)

        self.bAMtext = Tk.StringVar()
        self.bAMtext.set('Add Mask')
        self.buttonMask = Tk.Button(self.bFrame,textvariable=self.bAMtext,command=self.addMask,width=10)
        self.buttonMask.pack(side=Tk.LEFT)


    def deactivateButtons(self):
        if self.toolbar.mode!='':
            self.toolbar.zoom()
            self.toolbar.pan()
            self.toolbar.pan()
        if self.activeButton==self.buttonAG:
            self.bAGtext.set('Add Galaxy')
            self.canvas.mpl_disconnect(self.pressid)
        elif self.activeButton==self.buttonAL:
            self.bALtext.set('Add Lens')
            self.canvas.mpl_disconnect(self.pressid)
        elif self.activeButton==self.buttonAS:
            self.bAStext.set('Add Source')
            self.canvas.mpl_disconnect(self.pressid)
        elif self.activeButton==self.buttonMask:
            self.bAMtext.set('Add Mask')
            self.canvas.mpl_disconnect(self.pressid)
            self.canvas.mpl_disconnect(self.moveid)
            self.canvas.mpl_disconnect(self.releaseid)
        self.pressid = None
        self.releaseid = None
        self.activeButton = None


    def addMask(self,loaded=False):
        from matplotlib.patches import Rectangle
        if loaded and self.parent.mask is not None:
            import numpy
            y,x = numpy.where(self.parent.mask==1)
            x0,x1,y0,y1 = x.min(),x.max(),y.min(),y.max()
            self.rubberBox = Rectangle((x0,y0),x1-x0,y1-y0,fc='none',ec='w')
            self.a1.add_patch(self.rubberBox)
            self.canvas.draw()
            return
        if self.activeButton==self.buttonMask:
            self.deactivateButtons()
            return
        self.deactivateButtons()
        self.xmask = None
        def onPress(event):
            axes = event.inaxes
            if axes==self.a1:
                self.xmask = event.xdata
                self.ymask = event.ydata
            if self.rubberBox is not None:
                self.rubberBox.remove()
                self.rubberBox = None
        def onMove(event):
            if self.xmask is None:
                return
            axes = event.inaxes
            if axes==self.a1:
                x,y = event.xdata,event.ydata
                dx = x-self.xmask
                dy = y-self.ymask
                if self.rubberBox is None:
                    self.rubberBox = Rectangle((self.xmask,self.ymask),
                                                dx,dy,fc='none',ec='w')
                    self.a1.add_patch(self.rubberBox)
                else:
                    self.rubberBox.set_height(dy)
                    self.rubberBox.set_width(dx)
                self.canvas.draw()
        def onRelease(event):
            dy = int(self.rubberBox.get_height())
            dx = int(self.rubberBox.get_width())
            x0,y0 = int(self.xmask),int(self.ymask)
            x1,y1 = x0+dx,y0+dy
            self.parent.mask = self.parent.imgs[0]*0
            self.parent.mask[y0:y1,x0:x1] = 1
            self.parent.mask = self.parent.mask==1
            self.deactivateButtons()
        self.pressid = self.canvas.mpl_connect('button_press_event',onPress)
        self.moveid = self.canvas.mpl_connect('motion_notify_event',onMove)
        self.releaseid = self.canvas.mpl_connect('button_release_event',onRelease)
        self.bAMtext.set('Cancel')
        self.activeButton = self.buttonMask


    def showResid(self):
        if self.parent.models is None:
            self.a2.imshow(self.parent.img,origin='bottom',
                            interpolation='nearest')
            self.a3.cla()
            self.a3.set_xticks([])
            self.a3.set_yticks([])
            self.canvas.show()
            return
        models = self.parent.models
        imgs = self.parent.imgs
        nimgs = self.parent.nimgs
        if self.color is not None:
            if nimgs==2:
                b = imgs[0]-models[0]
                r = imgs[1]-models[1]
                g = (b+r)/2.
                resid = self.color.colorize(b,g,r)
                b = models[0]
                r = models[1]
                g = (b+r)/2.
                model = self.color.colorize(b,g,r,newI=True)
            else:
                b = imgs[0]-models[0]
                g = imgs[1]-models[1]
                r = imgs[2]-models[2]
                resid = self.color.colorize(b,g,r)
                b = models[0]
                g = models[1]
                r = models[2]
                model = self.color.colorize(b,g,r,newI=True)
        else:
            resid = imgs[0]-models[0]
            model = models[0]
            self.img3.set_clim([0.,model.max()])
        #self.a2.imshow(resid,origin='bottom',interpolation='nearest')
        #self.a3.imshow(model,origin='bottom',interpolation='nearest')
        self.img2.set_data(resid)
        self.img3.set_data(model)
        self.canvas.draw()

    def redrawSymbols(self):
        import objectMover
        if self.mover is not None:
            self.mover.remove()
        self.mover = objectMover.ObjMover(self.parent,self.a4,self.canvas)
