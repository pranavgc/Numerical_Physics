import numpy as np
import matplotlib.pyplot as plt
import sys
def f2(x1,x2,l,w):
	return -(2*l*w*x2+x1*(w**2))
def f1(x2):
	return x2
def r(l,w):
	t=np.linspace(0,100,10000)
	dt=t[2]-t[1]
	x1=np.zeros(len(t))
	x2=np.zeros(len(t))
	x1[0]=0.1
	x2[0]=0
	for i in range(len(t)-1):
		k11 = f1(x2[i])
		k12 = f2(x1[i],x2[i],l,w)
    		
		k21 = f1(x2[i]+k12*0.5*dt)
		k22 = f2(x1[i]+k11*dt*0.5,x2[i]+k12*0.5*dt,l,w)
    		
		k31 = f1(x2[i]+k22*0.5*dt)
		k32 = f2(x1[i]+k21*dt*0.5,x2[i]+k22*0.5*dt,l,w)
    		
		k41 = f1(x2[i]+k32*dt)
		k42 = f2(x1[i]+k31*dt,x2[i]+k32*dt,l,w)
		
		x1[i+1]=x1[i]+dt*(k11+2*k21*+2*k31+k41)/6
		x2[i+1]=x2[i]+dt*(k12+2*k22*+2*k32+k42)/6
	return x1,x2

'''t=np.linspace(0,10,10000)
a,b=r(1.5,0.5)
plt.plot(t,a)
plt.show()'''
c=np.arange(0,1.5,0.06)
c1=c[::-1]

for i in range(len(c1)):
        a,b=r(c1[i],0.5)
        for j in range(len(a)):
                if a[j]<0:
                        print 'lambda critical=',c1[i-1]
                        sys.exit()