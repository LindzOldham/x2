import indexTricks as iT

y,x = iT.coords((120,120))
y += 0.2
x += 0.2
y /= 20.
x /= 20.

q = 0.7
mu1 = 0.5*(y/x-x/y)
sbar = mu1 - 0.005*(1-q**2)/(2*x*y)
print sbar.min(),sbar.max()
