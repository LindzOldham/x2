import pyfits as py, numpy as np, pylab as pl

dir = '/data/ljo31b/EELs/esi/kinematics/inference/finals/'
s1 = 'chain/J0837_'
s2 = 'result/J0837_'
files = ['noextrap','blue','pickles','kurucz','stitch5000','stitch4000']

sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/J0837_wl.fits')[0].data
cond = [sciwave>np.log10(6000.), sciwave>np.log10(5200.),sciwave>np.log10(5200.),sciwave>np.log10(5200.),sciwave>np.log10(5000.),sciwave>np.log10(4000.)]

print r'model & $v_l$ & $\sigma_l$ & $v_s$ & $\sigma_s$ & $\bar{\Chi}^2$ \\\hline'
for i in range(len(files)):
    chain = np.loadtxt(dir+s1+files[i]+'.txt')
    lp = np.load(dir+s2+files[i])[0]
    lp = -2.*np.amax(lp)
    n = sciwave[cond[i]].size
    vl,sl,vs,ss = map(lambda v:(v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(chain, [16,50,84],axis=0)))
    print ' & $', '%.2f'%vl[0], '\pm', '%.2f'%vl[1], '$ & $', '%.2f'%sl[0], '\pm', '%.2f'%sl[1], '$ & $', '%.2f'%vs[0], '\pm', '%.2f'%vs[1], '$ & $', '%.2f'%ss[0], '\pm', '%.2f'%ss[1], '$ &', '%.2f.'%(lp/n), r' \\'
