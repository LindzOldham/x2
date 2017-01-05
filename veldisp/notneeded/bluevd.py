import pyfits as py,numpy as np,pylab as pl
import veltools as T
import special_functions as sf
from bluefitter import finddispersion,readresults
from scipy import ndimage,signal,interpolate
from math import sqrt,log10,log
import ndinterp
from tools import iCornerPlotter, gus_plotting as g

dir = '/home/mauger/python/vdfit/indous/' # move into my space soon
templates = ['102328_K3III.fits','163588_K2III.fits','107950_G5III.fits','124897_K1III.fits','168723_K0III.fits','111812_G0III.fits','148387_G8III.fits','188350_A0III.fits','115604_F2III.fits']

for i in range(len(templates)):
    templates[i] = dir+templates[i]


twave  = np.log10(T.wavelength(templates[0],0))
tscale = twave[1]-twave[0]
# science data resolution, template resolution

VGRID = 1.
light = 299792.458
ln10 = np.log(10.)

def getmodel(twave,tspec,tscale,smin=5.,smax=501):
    match = tspec.copy()
    # spectra are at higher resolution than templates, so don't need to degrade the latter
    disps = np.arange(smin,smax,VGRID)
    cube = np.empty((disps.size,twave.size))
    for i in range(disps.size):
        disp = disps[i]
        kernel = disp/(light*ln10*tscale)
        cube[i] = ndimage.gaussian_filter1d(match.copy(),kernel)
    X = disps.tolist()
    tx = np.array([X[0]]+X+[X[-1]])
    Y = twave.tolist()
    ty = np.array([Y[0]]+Y+[Y[-1]])
    return  (tx,ty,cube.flatten(),1,1) # ready to be ndinterpolated?

def run(zl,zs,fit=True,read=False,File=None,mask=None,lim=5200.,nfit=6.,bg='polynomial',nsrc=6,smooth='polynomial',bias=1e8):
    # Load in spectrum
    scispec = py.open('/data/ljo31b/EELs/esi/kinematics/J0837_spec.fits')[0].data
    varspec = py.open('/data/ljo31b/EELs/esi/kinematics/J0837_var.fits')[0].data
    sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/J0837_wl.fits')[0].data

    # cut nonsense data - nans at edges
    edges = np.where(np.isnan(scispec)==False)[0]
    start = edges[0]
    end = np.where(sciwave>np.log10(9500.))[0][0]
    scispec = scispec[start:end]
    varspec = varspec[start:end]
    datascale = sciwave[1]-sciwave[0] # 1.7e-5
    sciwave = 10**sciwave[start:end]

    zp = scispec.mean()
    scispec /= zp
    varspec /= zp**2

    # prepare the templates
    ntemps = len(templates)
    ntemp = 1
    result = []
    models = []
    t = []

    tmin = sciwave.min()
    tmax = sciwave.max()

    for template in templates:
        # Load the template
        file = py.open(template)
        tmpspec = file[0].data.astype(np.float64)
        tmpwave = T.wavelength(template,0)
        tmpspec /= tmpspec.mean()

        # do we need to smooth these? Presumably not, because our resolution is better
        twave = np.log10(tmpwave)
        tmpscale = twave[1]-twave[0]
        t.append(getmodel(twave,tmpspec,tmpscale)) 
    #return t
    if fit:
        result = finddispersion(scispec,varspec,t,tmpwave,datascale,tscale,np.log10(sciwave),zl,zs,nfit=nfit,outfile=File,mask=mask,lim=lim,bg=bg,nsrc=nsrc,smooth=smooth,bias=bias)
        return result
    elif read:
        result = readresults(scispec,varspec,t,tmpwave,datascale,tscale,np.log10(sciwave),zl,zs,nfit=nfit,infile=File,mask=mask,lim=lim,bg=bg,nsrc=nsrc,smooth=smooth,bias=bias)
        return result
    else:
        return

name = 'blue_final'
print name
result = run(0.4256,0.6411,fit=True,read=False,File = 'J0837_'+name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5200,nsrc=6,smooth='polynomial',bias=1e8)#,bias2=1e9)
result = run(0.4246,0.6411,fit=False,read=True,File = 'J0837_'+name,mask=np.array([[7580,7700],[6860,6900]]),nfit=6,bg='polynomial',lim=5200,nsrc=6,smooth='polynomial',bias=1e8)#,bias2=1e9)

#iCornerPlotter.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/J0837_bluepoly15_chain.txt,Crimson','/data/ljo31b/EELs/esi/kinematics/inference/J0837_bluepoly16_chain.txt,Blue','/data/ljo31b/EELs/esi/kinematics/inference/J0837_bluepoly17_chain.txt,Purple','/data/ljo31b/EELs/esi/kinematics/inference/J0837_bluepoly18_chain.txt,DarkOrange'])


lp,trace,dic,_ = result
print 'lp = ', np.amax(lp)
chain = g.changechain(trace,filename='/data/ljo31b/EELs/esi/kinematics/inference/J0837_'+name+'_chain')
g.triangle_plot(chain,burnin=125,axis_labels=['$v_l$',r'$\sigma_l$','$v_s$',r'$\sigma_s$'])
np.savetxt('/data/ljo31b/EELs/esi/kinematics/inference/J0837_'+name+'_chain.txt',chain[7500:,1:])

'''
iCornerPlotter.CornerPlotter(['/data/ljo31b/EELs/esi/kinematics/inference/J0837_bluepoly1_chain.txt,Crimson','/data/ljo31b/EELs/esi/kinematics/inference/J0837_bluepoly2_chain.txt,Blue','/data/ljo31b/EELs/esi/kinematics/inference/J0837_bluepoly3_chain.txt,Purple'])

# poly104 - poly10, not masking upper regions
# bluepoly103 - bluepoly10, not masking upper regions

#'/data/ljo31b/EELs/esi/kinematics/inference/J0837_poly202_chain.txt,Crimson','/data/ljo31b/EELs/esi/kinematics/inference/J0837_poly252_chain.txt,Purple','/data/ljo31b/EELs/esi/kinematics/inference/J0837_legendre202_chain.txt,SeaGreen','/data/ljo31b/EELs/esi/kinematics/inference/J0837_poly203_chain.txt,LightGray'])
pl.show()


Hi Judith,

I'm writing to ask about the room you're advertising in Newnham, and whether it is still available/we might discuss it further. I'm a PhD student at the Institute of Astronomy here in Cambridge, and am currently living in college-owned accommodation but am really looking to move out and get some more independence. Also...the main reason I want to move elsewhere is because I really want to get a hamster (which College doesn't allow) -- so when I saw your advert, I thought we might be a good match!! As a housemate, I am very quiet, enjoy a chat in the evenings or at the weekend and the occasional film or outing, but also tend to spend a lot of time either out at work or doing things in my room - would this work for you, or are you looking for someone very sociable? (As I say, I'm not UNsociable, I just want to make sure I'm painting a clear picture of myself!). In my spare time, I enjoy triathlon training and reading (which I also notice overlap quite a lot with your hobbies!) - and when I get my hamster... Anyway, it would be great to hear more from you if you think we might get on! My mobile number is 07818 646772 if that's an easier way to contact me.

Hoping to hear from you soon!

Lindsay

'''
