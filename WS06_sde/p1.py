#Coded in Python 2.7.15
import numpy as np
import matplotlib.pyplot as plt
	 
def g(dt):
	
	def f1(x1,t1,z1,m1,si1):
	 	return m1*x1*t1+si1*x1*z1*t1**0.5

	def f0(x0,t0,n0,m0,si0):
		return x0*np.exp(m0*t0-si0**2*t0/2.0+si0*n0)

	t=np.arange(0,35,dt)#time

	z=np.random.randn(len(t))#random numbers

        #constants
	m=0.25
	si=1.2

	x=np.zeros(len(z))#for storing simulated data
	y=np.zeros(len(z))#for storing expected data
	
	x[0]=1
	y[0]=x[0]

	n=np.zeros(len(z))#for storing eta(t)

        #Code for integrating simulated data
	for i in range(len(z)-1):
		n[i+1]=n[i]+z[i]*dt**1.5#calculating eta(t)
		x[i+1]=x[i]+f1(x[i],dt,z[i+1],m,si)#integrating simulated data
		y[i+1]=f0(x[0],t[i+1],n[i+1],m,si)#calculating expected data
	#print x
	#print y

	#this code below is plot for (a) part of question

	'''plt.plot(t,x,label='simulated data')
	
	plt.plot(t,y,label='expected data')
	plt.xlabel('time')
	plt.ylabel('x')
	plt.legend()
	plt.show()'''

        #rms error
	r=np.sqrt(np.sum(np.square(y-x))/len(x))

	return r

#Code for part (b) 
d=np.linspace(0.1,0.5,10)#dt array
a=np.zeros(len(d))#array for storing error
for j in range(len(d)):
	a[j]=g(d[j])
#Plots
plt.plot(d,a,label='error')
plt.xlabel('dt')
plt.ylabel('error')
plt.legend()
plt.show()

