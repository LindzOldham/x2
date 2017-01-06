import pyfits as py, pylab as pl, numpy as np

files = ['/data/mauger/EELs/esi/jan2015/e150123_00'+str(i)+'.fits' for i in range(33,54)]

for file in files:
    header = py.open(file)[0].header
    print '<p>',file[-7:-5],header['TARGNAME'], '%.2f'%header['ROTPOSN']
