from linslens.EELsImages import *# import EELsImages

def my_import(name):
    m = __import__(name)
    print m
    for n in name.split(".")[1:]:
        print n
        m = getattr(m, n)
    return m
