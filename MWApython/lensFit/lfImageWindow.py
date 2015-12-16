from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import Tkinter as Tk
import Tix
from Tkconstants import *
from math import log10

def absPower(img,pow):
    import numpy
    return numpy.sign(img)*numpy.abs(img)**pow


class ImageWindow:
    def __init__(self,parent):
        self.parent = parent
        self.visible = False
        self.color = None
        self.draw_window()

    def draw_window(self):
        stretch = self.parent.imgScaling

        top = Tix.Toplevel()
        top.resizable(0,0)

        monW = top.winfo_screenmmwidth()/(25.4)
        monH = top.winfo_screenmmheight()/(25.4)

        if self.parent.img.ndim==3:
            imshape = self.parent.img[:,:,0].shape
        else:
            imshape = self.parent.img.shape

        if imshape[0]>imshape[1]:
            # Set height to be 2/3 of screen height
            fheight = monH*2/3.
            fwidth = 3*fheight*imshape[1]/imshape[0]
            if fwidth>monW*2/3.:
                factor = (monW*2/3.)/fwidth
                fheight *= factor
                fwidth *= factor
        else:
            fwidth = monW*2/3.
            fheight = fwidth*imshape[0]/imshape[1]/3.

        f1 = Figure((fwidth,fheight),dpi=96)
        a1 = f1.add_axes([0,0,0.331,1.])
        self.img1 = a1.imshow(absPower(self.parent.img,stretch),origin='bottom',interpolation='nearest')
        a1.set_xticks([])
        a1.set_yticks([])
        xlim = a1.get_xlim()
        ylim = a1.get_ylim()

        a2 = f1.add_axes([0.3345,0,0.331,1.])
        self.img2 = a2.imshow(absPower(self.parent.resid,stretch),origin='bottom',interpolation='nearest')
        a2.set_xlim(xlim)
        a2.set_ylim(ylim)
        a2.set_xticks([])
        a2.set_yticks([])

        a3 = f1.add_axes([0.669,0,0.331,1.,])
        self.img3 = a3.imshow(absPower(self.parent.model,stretch),origin='bottom',interpolation='nearest')
        a3.set_xlim(xlim)
        a3.set_ylim(ylim)
        a3.set_xticks([])
        a3.set_yticks([])


        self.img2.set_clim(self.img1.get_clim())
        self.img3.set_clim(self.img1.get_clim())

        self.scaleVar = Tk.DoubleVar()
        self.scaleVar.set(log10(stretch))
        scaleFrame = Tk.Frame(top)
        scaleFrame.pack(side=RIGHT,fill=BOTH)
        label = Tk.Label(scaleFrame,text='Adjust\nContrast')
        label.pack(side=TOP,fill=X)
        h1 = top.winfo_reqheight()
        h2 = label.winfo_reqheight()
        h = h1-h2*2
        scaler = Tk.Scale(scaleFrame,orient=VERTICAL,command=self.change_scaling,showvalue=0,from_=-1.,to=1.,variable=self.scaleVar,resolution=0.05)
        scaler.pack()
        scaler.place(anchor=CENTER,relx=0.5,rely=0.5,relheight=h/float(h1))

        canvas = FigureCanvasTkAgg(f1,master=top)
        canvas.show()
        canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=1)
        toolbar = NavigationToolbar2TkAgg(canvas,top)
        toolbar.update()
        canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
        self.visible = True
        self.toolbar = toolbar


        self.f1 = f1
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.canvas = canvas
        self.top = top
        self.top.protocol('WM_DELETE_WINDOW',self.close_window)

    def close_window(self):
        self.top.destroy()
        self.parent.gui.deactivateButtons()
        self.visible = False

    def update_window(self):
        if self.visible==False:
            self.draw_window()
            return
        self.img2.set_data(absPower(self.parent.resid,self.parent.imgScaling))
        self.img3.set_data(absPower(self.parent.model,self.parent.imgScaling))
#        if self.parent.model.ndim==2:
#            self.img3.set_clim([0.,self.parent.model.max()])
        self.canvas.draw()

    def change_scaling(self,event):
        stretch = 10**self.scaleVar.get()
        self.parent.imgScaling = stretch
        self.img1 = self.a1.imshow(absPower(self.parent.img,stretch),origin='bottom',interpolation='nearest')
#        self.a1.set_xticks([])
#        self.a1.set_yticks([])
        self.img2.set_data(absPower(self.parent.resid,stretch))
        self.img3.set_data(absPower(self.parent.model,stretch))
        self.img2.set_clim(self.img1.get_clim())
        self.img3.set_clim(self.img1.get_clim())
        self.canvas.draw()


