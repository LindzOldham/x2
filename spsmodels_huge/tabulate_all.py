import numpy as np

names =  np.load('/data/ljo31/Lens/LensParams/Alambda_hst.npy')[()].keys()
names.sort()

m1 = np.load('/data/ljo31/Lens/LensParams/F606W_rightredshifts_data.npy')
m2 = np.load('/data/ljo31/Lens/LensParams/F606W_rightredshifts_model.npy')
m3 = np.load('/data/ljo31/Lens/LensParams/F606W_redshift0_model.npy')
m4 = np.load('/data/ljo31/Lens/LensParams/V_redshift0_model.npy')

for ii in range(m1.size):
    print names[ii], '&', '%.2f'%m1[ii], '&', '%.2f'%m2[ii], '&', '%.2f'%m3[ii], '&', '%.2f'%m4[ii], r'\\'


m2 = np.load('/data/ljo31/Lens/LensParams/F606W_rightredshifts_model_marginalised.npy')
logI = VV-8.5
dlogI = np.mean((VVup-VV,VV-VVlo),0)
