import VDfit
import pyfits as py
import numpy

sigsci = lambda wave: 20.26
t1 = VDfit.INDOUS(sigsci)
t2 = VDfit.PICKLES(sigsci)
linwave = numpy.linspace(3400,8000,100000)#t1.wave
#scispec = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/J0837_ap_1.00_spec_lens.fits')[0].data
#sciwave = py.open('/data/ljo31b/EELs/esi/kinematics/apertures/final/J0837_ap_1.00_wl_lens.fits')[0].data
#linwave=10**sciwave
t1.getSpectra(linwave,0.5,200.)
t2.getSpectra(linwave,0.,0.)
#wave = np.arange(3400.,8000.,0.1)
