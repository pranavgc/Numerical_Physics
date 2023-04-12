#Coded in python 3.7.3
import numpy as np
import matplotlib.pyplot as plt
def d2(x1,x3):
	return -x1+(2*x3**2)/float(x1)
def d(x2):
	return x2
def F(c):
	x=np.linspace(-1.0,1.0,100)
	dx=x[1]-x[0]
	y=np.zeros(len(x))
	p=np.zeros(len(x))
	y[0]=1/(np.exp(1)+np.exp(-1))
	p[0]=c
	for i in range(len(x)-1):
		k1=d2(y[i],p[i])
		u1=d(p[i])
		
		k2=d2(y[i]+dx/2.0*u1,p[i]+dx/2.0*k1)
		u2=d(p[i]+dx/2.0*k1)
		
		k3=d2(y[i]+dx/2.0*u2,p[i]+dx/2.0*k2)
		u3=d(p[i]+dx/2.0*k2)
		
		k4=d2(y[i]+dx*u3,p[i]+dx*k3)
		u4=d(p[i]+dx*k3)
		
		p[i+1]=p[i]+dx/6.0*(k1+2*k2+2*k3+k4)
		y[i+1]=y[i]+dx/6.0*(u1+2*u2+2*u3+u4)
	
	return y[-1],y


v1=0.2
v2=0.3
x=np.linspace(-1.0,1.0,100)

while np.abs(F(v1)[0]-(1/(np.exp(1)+np.exp(-1))))>0.001:
	a1,b1=F(v1)
	a2,b2=F(v2)
	v3=v2-((a2-1/(np.exp(1)+np.exp(-1)))*(v2-v1)/(float(a2-a1)))
	v1=v2
	v2=v3
else:
	y1=F(v1)[1]
plt.plot(x,y1,label="Numerical solution")
plt.plot(x,1/(np.exp(x)+np.exp(-x)),"*",label="Actual solution")
plt.xlabel("x")
plt.ylabel("y=1/(np.exp(x)+np.exp(-x))")
plt.legend()
plt.show()