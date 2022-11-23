import os
import numpy

a = numpy.loadtxt('pace.txt')
print(a)
print(a.shape)
print(a[0][7:(7 + 1)])