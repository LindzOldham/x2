import matplotlib as mpl
from numpy import pi,arctan2

class ObjMover:
    def __init__(self,parent,ax,canvas):
        self.parent = parent
        self.ax = ax
        self.canvas = canvas
        self.markers = None
        self.objs = None
        self.addComponents()
        self.picked = None
        self.activePos = None
        self.opos = None
        self.mindex = None
        if self.markers is not None:
            self.connectPicker()
            self.canvas.draw()

    def addComponents(self):
        self.markers = []
        self.objs = []
        self.createEllipse(self.parent.galaxyManager,'r')
        self.createEllipse(self.parent.srcManager,'b')
        self.createEllipse(self.parent.lensManager,'k')

    def createEllipse(self,manager,color):
        for name,obj in manager.objs.iteritems():
            x,y = obj.pars['x']['value'],obj.pars['y']['value']
            if color=='k':
                r = obj.pars['b']['value']
            else:
                r = obj.pars['re']['value']
            q,pa = obj.pars['q']['value'],obj.pars['pa']['value']
            height = 2*r*q**0.5
            width = 2*r/q**0.5
            ellipse = mpl.patches.Ellipse((x,y),width,height,pa,fc='none',
                                            ec=color)
            ellipse.manager = manager
            self.ax.add_patch(ellipse)
            self.markers.append(ellipse)
            self.objs.append(obj)

    def connectPicker(self):
        self.pressid = self.canvas.mpl_connect('button_press_event',
                                                self.onPress)

    def connect(self):
        self.moveid = self.canvas.mpl_connect('motion_notify_event',
                            self.onMotion)
        self.offid = self.canvas.mpl_connect('button_release_event',
                            self.onRelease)

    def disconnect(self):
        if self.picked is not None:
            self.canvas.mpl_disconnect(self.moveid)
            self.canvas.mpl_disconnect(self.offid)

    def onPress(self,event):
        if event.inaxes!=self.ax:
            return

        if self.canvas.toolbar.mode!='':
            if event.button==2:
                self.canvas.toolbar.zoom()
                self.canvas.toolbar.pan()
                self.canvas.toolbar.pan()

        ex,ey = event.xdata,event.ydata
        if self.picked is None:
            # Figure out which patch was clicked
            offsets = []
            for m in self.markers:
                if m.contains_point((event.x,event.y)):
                    x,y = m.center
                    offsets.append(((x-ex)**2+(y-ey)**2))
                else:
                    offsets.append(1e12)
            i = offsets.index(min(offsets))
            if offsets[i]>1e11:
                return
            self.picked = self.markers[i]
            self.mindex = i
            self.connect()

        if event.button>1:
            ax,ay = self.picked.center
            x,y = event.xdata,event.ydata
            r = ((ax-x)**2+(ay-y)**2)**0.5
            q = self.picked.height/self.picked.width
            s = (self.picked.height*self.picked.width)**0.5
            self.startR = r
            self.startH = self.picked.height
            self.startW = self.picked.width
            self.startA = self.picked.angle
            self.startQ = q
            self.startS = s
            angle = arctan2(ay-y,ax-x)*180/pi+90.
            self.picked.angle = angle
        if event.button==3:
            self.picked.height = 2*self.startR*self.startQ**0.5
            self.picked.width = 2*self.startR/self.startQ**0.5
        self.activePos = event.xdata,event.ydata
        self.opos = self.picked.center

    def onMotion(self,event):
        if event.inaxes!=self.ax or self.picked is None:
            return
        if event.button==1:
            ax,ay = self.activePos
            dx,dy = ax-event.xdata,ay-event.ydata
            x,y = self.opos
            self.picked.center = x-dx,y-dy
            self.objs[self.mindex].pars['x']['value'] = x-dx
            self.objs[self.mindex].pars['y']['value'] = y-dy
        elif event.button==2:
            ax,ay = self.opos
            x,y = event.xdata,event.ydata
            r = ((ax-x)**2+(ay-y)**2)**0.5
            q = r/self.startR
            if q>20:
                q = 20
            if q<0.05:
                q = 0.05
            self.picked.height = self.startH*q**0.5
            self.picked.width = self.startW/q**0.5
            ax,ay = self.opos
            self.picked.angle = arctan2(ay-y,ax-x)*180/pi+90.
            if self.picked.get_ec()=='k':
                self.objs[self.mindex].pars['b']['value'] = r
            else:
                self.objs[self.mindex].pars['re']['value'] = r
            self.objs[self.mindex].pars['pa']['value'] = self.picked.angle
        elif event.button==3:
            ax,ay = self.opos
            x,y = event.xdata,event.ydata
            q = self.startQ
            r = ((ax-x)**2+(ay-y)**2)**0.5
            self.picked.height = 2*r*q**0.5
            self.picked.width = 2*r/q**0.5
            self.picked.angle = arctan2(ay-y,ax-x)*180/pi+90.
            angle = self.picked.angle
            if q>1:
                q = 1./q
                if angle>90.:
                    angle -= 90.
                else:
                    angle += 90.
            self.objs[self.mindex].pars['q']['value'] = q
            self.objs[self.mindex].pars['pa']['value'] = angle
        if self.picked.manager.tableVisible:
            self.picked.manager.table.drawTable()
        #if self.picked.get_ec()=='r':
        #    if self.parent.
        self.canvas.draw()

    def onRelease(self,event):
        self.disconnect()
        self.mindex = None
        self.picked = None
        self.opos = None
        self.parent.fitLight(False)
        self.canvas.draw()

    def remove(self):
        for m in self.markers:
            m.remove()
            del m
        self.disconnect()
