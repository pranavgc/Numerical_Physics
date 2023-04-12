import numpy as np
import matplotlib.pyplot as plt
def d2(x1,x3):
	return -x1+(2*x3**2)/float(x1)
def d(x2):
	return x2
def F(c):
	x=np.linspace(-1.0,1.0,100)
	dx=x[1]-x[0]
	dx2=-dx
	y=np.zeros(int(len(x)/2))
	p=np.zeros(int(len(x)/2))
	y[0]=1/(np.exp(1)+np.exp(-1))
	p[0]=c

	y2=np.zeros(int(len(x)/2))
	p2=np.zeros(int(len(x)/2))
	y2[0]=1/(np.exp(1)+np.exp(-1))
	p2[0]=-c
	for i in range((int(len(x)/2))-1):
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


		k12=d2(y2[i],p2[i])
		u12=d(p2[i])
		
		k22=d2(y2[i]+dx2/2.0*u12,p2[i]+dx2/2.0*k12)
		u22=d(p2[i]+dx2/2.0*k12)
		
		k32=d2(y2[i]+dx2/2.0*u22,p2[i]+dx2/2.0*k22)
		u32=d(p2[i]+dx2/2.0*k22)
		
		k42=d2(y2[i]+dx2*u32,p2[i]+dx2*k32)
		u42=d(p2[i]+dx2*k32)
		
		p2[i+1]=p2[i]+dx2/6.0*(k12+2*k22+2*k32+k42)
		y2[i+1]=y2[i]+dx2/6.0*(u12+2*u22+2*u32+u42)
	z=np.array([ y[-4],y2[-4],y,y2])
	return z


v1=0.2
v2=0.32
z1=F(0.2)
x=np.linspace(-1.0,1.0,100)
f=1

while (np.abs(z1[0] - z1[1]))>0.001:
	z1=F(v1)
	z2=F(v2)
	f=f+1
	v3=v2-((z2[0]-z2[1])*(v2-v1)/(float((z2[0]-z2[1])-(z1[0]-z1[1]))))
	v1=v2
	v2=v3
else:
	s=z1[3]
	#print(s,s[::-1])
	y1=np.concatenate((z1[2],s[::-1]))

plt.plot(x,y1,label="Numerical solution")
plt.plot(x,1/(np.exp(x)+np.exp(-x)),"*",label="Actual solution")
plt.xlabel("x")
plt.ylabel("y=1/(np.exp(x)+np.exp(-x))")
plt.legend()
plt.show()