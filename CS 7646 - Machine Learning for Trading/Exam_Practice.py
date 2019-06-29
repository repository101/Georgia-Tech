
# Question 1
import numpy as np
A = np.ones((3,3))
w = np.array([0.0, 0.1, 0.2])
print (A*w).sum()
print (A*w).sum(axis=0)
print (A*w).sum(axis=1)
temp = (A*w)
print ""

# Question 2
a = 3
print "a = {}".format(a)
b = a
print "b = {}".format(b)
a = 2
print "a = {}\n" \
      "b = {}".format(a, b)
temp = b * a
print b * a
print "temp = {}".format(temp)
print ""

# Question 3
a = np.random.uniform(size=(3, 2))
print "a = \n{}".format(a)
temp = a[1, :]
print "temp = \n{}".format(temp)
b = a/a[1, :]
print "b = \n{}".format(b)
print ""