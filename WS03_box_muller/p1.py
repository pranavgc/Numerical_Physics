import numpy as np
import matplotlib.pyplot as plt
import collections as co 
N=10000
x=np.sort(np.random.rand(10000))
dx=(np.max(x)-np.min(x))/N
y=(x*N)
y=y.astype(int)
c=[]
z=list(y)
for i in range(N):
	z1=z.count(i)
	c.append(z1)
	
w=np.arange(np.min(x),np.max(x),dx)
g=np.asarray(c)
g1=np.sum(g*dx)
w=w[:len(g)]
v=np.sum(g/g1)*dx
print 'area under the graph',v
#print g1
plt.plot(w,g/g1)
#plt.hist(x,N)
plt.legend()
plt.title('Normalized Uniform Distribution')
plt.xlabel('X')
plt.ylabel('P(X)')

plt.show()




