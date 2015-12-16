import Tkinter as Tk

class ParTable:
    def __init__(self,parent):
        self.parent = parent
        self.Table = None

    def drawTable(self,root):
        objs = self.parent.objs
        pars = self.parent.pars
        if len(objs)==0:
            return

#        if self.Table is not None:
#            for c in self.Table.children.keys():
#                self.Table.children[c].destroy()
#        else:
#            self.Table = Tk.Toplevel()
        self.Table= root
        buttons = []
        spacers = []
        keys = objs.keys()
        keys.sort()
        for key in keys:
            obj = objs[key]
            f = Tk.Frame(self.Table,width=420)
            f.pack(side=Tk.LEFT)

            row0 = Tk.Frame(f)
            row0.grid(columnspan=5)
            lname = Tk.Label(row0,text=key)
            lname.pack(side=Tk.LEFT)
            buttons.append(Tk.Button(row0,text='Delete',command=lambda i=key: self.deleteObject(i)).pack(side=Tk.LEFT))
            row = 1
            for parkey in pars:
                r = TableRow(f,parkey,row,obj.pars[parkey],self.parent)
                row += 1
            B = Tk.Button(self.Table,bitmap='gray12',width=1,height=200)
            B.configure(state='disabled')
            B.pack(side=Tk.LEFT)
            spacers.append(B)
        spacers[-1].destroy()
#        self.Table.protocol("WM_DELETE_WINDOW",self.closeTable)
#        self.parent.tableVisible = True

    def closeTable(self):
        self.Table.destroy()
        self.Table = None
        self.parent.tableClosed()

    def deleteObject(self,name):
        self.parent.deleteObject(name)


class TableEntry:
    def __init__(self,frame,obj,objTab):
        self.obj = obj
        self.frame = frame
        self.objTab = objTab
        self.rows = []
        self.drawTable()

    def drawTable(self):
        row0 = Tk.Frame(self.frame,width=self.frame.winfo_width())
        row0.grid(columnspan=6)
        lname = Tk.Label(row0,text=self.obj.name)
        lname.pack(side=Tk.LEFT)
        button = Tk.Button(row0,text='Delete',command=self.delete)
        button.pack(side=Tk.LEFT)
        row = 1
        for key in self.obj.keys:
            self.rows.append(TableRow(self.frame,key,row,self.obj.pars[key],self.obj.manager.parent))
            row += 1

    def delete(self):
        self.obj.manager.deleteObject(self.obj.name)
        self.destroy()
        self.objTab.removeModel()

    def destroy(self):
        children = self.frame.winfo_children()
        for child in children:
            child.destroy()

    def update(self):
        self.destroy()
        self.drawTable()


class TableRow:
    def __init__(self,frame,label,row,obj,parent,width=5):
        self.obj = obj
        self.frame = frame
        self.parent = parent

        self.label = Tk.Label(self.frame,text=label,width=3)
        self.label.grid(row=row,column=0)

        self.valueVar = Tk.DoubleVar()
        self.valueVar.set(self.obj['value'])
        self.valueBox = Tk.Entry(self.frame,textvariable=self.valueVar,width=width)
        self.valueBox.grid(row=row,column=1)

        self.typeVar = Tk.StringVar()
        self.typeVar.set(self.obj['type'])
        self.typeBox = Tk.OptionMenu(self.frame,self.typeVar,'constant','uniform','normal',command=self.changeType)
        self.typeBox.grid(row=row,column=2)

        self.var1 = Tk.DoubleVar()
        self.var2 = Tk.DoubleVar()
        if self.obj['type']=='normal':
            self.var1.set(self.obj['mean'])
            self.var2.set(self.obj['sigma'])
        else:
            self.var1.set(self.obj['lower'])
            self.var2.set(self.obj['upper'])
        self.var1Box = Tk.Entry(self.frame,textvariable=self.var1,width=width)
        self.var2Box = Tk.Entry(self.frame,textvariable=self.var2,width=width)
        self.var1Box.grid(row=row,column=3)
        self.var2Box.grid(row=row,column=4)
        if self.obj['type']=='constant':
            self.disableEntry()
        self.valueBox.bind('<FocusOut>',self.textUpdate)
        self.var1Box.bind('<FocusOut>',self.textUpdate)
        self.var2Box.bind('<FocusOut>',self.textUpdate)


    def textUpdate(self,event):
        v = event.widget.get()
        try:
            v = float(v)
            self.updateObj()
        except:
            if event.widget==self.valueBox:
                self.valueVar.set(self.obj['value'])
            elif event.widget==self.var1Box:
                if self.obj['type']=='uniform':
                    self.var1.set(self.obj['lower'])
                else:
                    self.var1.set(self.obj['mean'])
            elif event.widget==self.var2Box:
                if self.obj['type']=='uniform':
                    self.var1.set(self.obj['upper'])
                else:
                    self.var1.set(self.obj['sigma'])

    def changeType(self,event=None,doCheck=True):
        ptype = self.typeVar.get()
        if doCheck and ptype==self.obj['type']:
            return
        if ptype=='constant':
            self.disableEntry()
        else:
            self.enableEntry()
        if ptype=='normal':
            self.var1.set(self.obj['mean'])
            self.var2.set(self.obj['sigma'])
        elif ptype=='uniform':
            self.var1.set(self.obj['lower'])
            self.var2.set(self.obj['upper'])
        self.updateObj()

    def disableEntry(self):
        self.var1Box.configure(state='readonly')
        self.var2Box.configure(state='readonly')

    def enableEntry(self):
        self.var1Box.configure(state='normal')
        self.var2Box.configure(state='normal')

    def updateObj(self,showUpdate=True):
        isUpdated = False
        ptype = self.typeVar.get()
        value = float(self.valueVar.get())
        if self.obj['value']!=value:
            isUpdated = True
        if ptype=='uniform':
            v = value
            l = float(self.var1.get())
            u = float(self.var2.get())
            if u<l:
                u = l
                self.var2.set(u)
            if v<l:
                l = v
                self.var1.set(l)
            elif v>u:
                u = v
                self.var2.set(u)
            self.obj['lower'] = l
            self.obj['upper'] = u
        elif ptype=='normal':
            self.obj['mean'] = float(self.var1.get())
            self.obj['sigma'] = float(self.var2.get())
        self.obj['value'] = value
        self.obj['type'] = ptype
        if isUpdated and showUpdate:
            #self.parent.gui.showImg()#.updateDisplay()
            self.parent.fitLight()
