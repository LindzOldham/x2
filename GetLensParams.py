import pyfits as py, numpy as np

def MakeDict(result):
    lp,trace,dic,_=result
    ftrace = trace[:,0].reshape((trace.shape[0]*trace.shape[2],trace.shape[3]))
    upperlower = map(lambda v: (v[0],v[1],v[2]),zip(*np.percentile(ftrace,[16,50,84],axis=0)))

    l,u,d = [], [], []
    for key in dic.keys():
        f = dic[key][:,0].reshape((trace.shape[0]*trace.shape[2]))
        lo,med,up = np.percentile(f,50)-np.percentile(f,16), np.percentile(f,50), np.percentile(f,84)-np.percentile(f,50) 
        d.append((key,med))
        l.append((key,lo))
        u.append((key,up))
    #print key, '$', '%.2f'%np.percentile(f,50), '_{-', '%.2f'%np.percentile(f,16), '}^{+', '%.2f'%np.percentile(f,84), '}$'

    Ddic = dict(d)                    
    Ldic = dict(l)
    Udic = dict(u)
    return Ddic,Ldic,Udic

def PrintTable(Ddic,Ldic,Udic):
    print r'\begin{table}[H]'
    print r'\centering'
    print r'\begin{tabular}{|c|cccccc|}\hline'
    print r' object & x & y & re & n & pa & q \\\hline'
    print 'source 1 & $', '%.2f'%(Ddic['Source 1 x']+Ddic['Lens 1 x']), '_{-', '%.2f'%Ldic['Source 1 x'],'}^{+','%.2f'%Udic['Source 1 x'], '}$ & $', '%.2f'%(Ddic['Source 1 y']+Ddic['Lens 1 y']),'_{-', '%.2f'%Ldic['Source 1 y'],'}^{+', '%.2f'%Udic['Source 1 y'], '}$ & $', '%.2f'%Ddic['Source 1 re'],'_{-', '%.2f'%Ldic['Source 1 re'],'}^{+', '%.2f'%Udic['Source 1 re'], '}$ & $', '%.2f'%Ddic['Source 1 n'],'_{-', '%.2f'%Ldic['Source 1 n'],'}^{+', '%.2f'%Udic['Source 1 n'], '}$ & $','%.2f'%Ddic['Source 1 pa'],'_{-', '%.2f'%Ldic['Source 1 pa'],'}^{+', '%.2f'%Udic['Source 1 pa'], '}$ & $','%.2f'%Ddic['Source 1 q'],'_{-', '%.2f'%Ldic['Source 1 q'],'}^{+', '%.2f'%Udic['Source 1 q'], '}$',r'\\'
    ###
    print 'galaxy 1 & $', '%.2f'%Ddic['Galaxy 1 x'], '_{-', '%.2f'%Ldic['Galaxy 1 x'],'}^{+','%.2f'%Udic['Galaxy 1 x'], '}$ & $', '%.2f'%Ddic['Galaxy 1 y'],'_{-', '%.2f'%Ldic['Galaxy 1 y'],'}^{+', '%.2f'%Udic['Galaxy 1 y'], '}$ & $', '%.2f'%Ddic['Galaxy 1 re'],'_{-', '%.2f'%Ldic['Galaxy 1 re'],'}^{+', '%.2f'%Udic['Galaxy 1 re'], '}$ & $', '%.2f'%Ddic['Galaxy 1 n'],'_{-', '%.2f'%Ldic['Galaxy 1 n'],'}^{+', '%.2f'%Udic['Galaxy 1 n'], '}$ & $','%.2f'%Ddic['Galaxy 1 pa'],'_{-', '%.2f'%Ldic['Galaxy 1 pa'],'}^{+', '%.2f'%Udic['Galaxy 1 pa'], '}$ & $','%.2f'%Ddic['Galaxy 1 q'],'_{-', '%.2f'%Ldic['Galaxy 1 q'],'}^{+', '%.2f'%Udic['Galaxy 1 q'], '}$',r'\\'
    ###
    print 'galaxy 2 & $', '%.2f'%Ddic['Galaxy 2 x'], '_{-', '%.2f'%Ldic['Galaxy 2 x'],'}^{+','%.2f'%Udic['Galaxy 2 x'], '}$ & $', '%.2f'%Ddic['Galaxy 2 y'],'_{-', '%.2f'%Ldic['Galaxy 2 y'],'}^{+', '%.2f'%Udic['Galaxy 2 y'], '}$ & $', '%.2f'%Ddic['Galaxy 2 re'],'_{-', '%.2f'%Ldic['Galaxy 2 re'],'}^{+', '%.2f'%Udic['Galaxy 2 re'], '}$ & $', '%.2f'%Ddic['Galaxy 2 n'],'_{-', '%.2f'%Ldic['Galaxy 2 n'],'}^{+', '%.2f'%Udic['Galaxy 2 n'], '}$ & $','%.2f'%Ddic['Galaxy 2 pa'],'_{-', '%.2f'%Ldic['Galaxy 2 pa'],'}^{+', '%.2f'%Udic['Galaxy 2 pa'], '}$ & $','%.2f'%Ddic['Galaxy 2 q'],'_{-', '%.2f'%Ldic['Galaxy 2 q'],'}^{+', '%.2f'%Udic['Galaxy 2 q'], '}$',r'\\'
    ###
    print 'lens 1 & $', '%.2f'%Ddic['Lens 1 x'], '_{-', '%.2f'%Ldic['Lens 1 x'],'}^{+','%.2f'%Udic['Lens 1 x'], '}$ & $', '%.2f'%Ddic['Lens 1 y'],'_{-', '%.2f'%Ldic['Lens 1 y'],'}^{+', '%.2f'%Udic['Lens 1 y'], '}$ & $', '%.2f'%Ddic['Lens 1 b'],'_{-', '%.2f'%Ldic['Lens 1 b'],'}^{+', '%.2f'%Udic['Lens 1 b'], '}$ & $', '%.2f'%Ddic['Lens 1 eta'],'_{-', '%.2f'%Ldic['Lens 1 eta'],'}^{+', '%.2f'%Udic['Lens 1 eta'], '}$ & $','%.2f'%Ddic['Lens 1 pa'],'_{-', '%.2f'%Ldic['Lens 1 pa'],'}^{+', '%.2f'%Udic['Lens 1 pa'], '}$ & $','%.2f'%Ddic['Lens 1 q'],'_{-', '%.2f'%Ldic['Lens 1 q'],'}^{+', '%.2f'%Udic['Lens 1 q'], '}$',r'\\\hline'
    ###
    print r'\end{tabular}'
    print r'\caption{', 'shear = $', '%.2f'%Ddic['extShear'], '_{-', '%.2f'%Ldic['extShear'],'}^{+','%.2f'%Udic['extShear'], '}$ , shear pa = $',  '%.2f'%Ddic['extShear PA'], '_{-', '%.2f'%Ldic['extShear PA'],'}^{+','%.2f'%Udic['extShear PA'], '}$}'
    print r'\end{table}'
