import numpy as np
import matplotlib.pyplot as plt

def G(n):

	def F(z):

		def d2(y2):
			return -4.0*y2

		def d(y):
			return y



		x=np.linspace(0,np.pi/4.0,n)
		dx=x[1]-x[2]
		y=np.zeros(len(x))
		v=np.zeros(len(x))
	#y2=np.zeros([len(x)])
	#v2=np.zeros([len(x)])
		y[0]=-2
		v[0]=z
		

		for i in range(len(x)-1):
			k11=d2(y[i])
			k12=d(v[i])

			k21=d2(y[i]+k12*dx*0.5)
			k22=d(v[i]+k11*dx*0.5)

			k31=d2(y[i]+k22*dx*0.5)
			k32=d(v[i]+k21*dx*0.5)

			k41=d2(y[i]+k32*dx)
			k42=d(v[i]+k31*dx)
			
			v[i+1]=v[i]+(k11+2*k21+2*k31+k41)*dx/6.0
			y[i+1]=y[i]+(k12+2*k22+2*k32+k42)*dx/6.0

		return y[len(x)-1],y

	#print x,np.pi/4
	#print y



	x=np.linspace(0,np.pi/4.0,n)
	dx=x[2]-x[1]
	v=1
	v2=-10
	f=0
	a,b=F(v)
	yz=10.0
	while np.abs(a-yz)>0.0001:
		f=f+1
		a,b=F(v)
		a2,b2=F(v2)
		
			
			
		v3=v2-(a2-yz)*(v2-v)/(a2-a)
		v=v2
		v2=v3
		
		#plt.plot(x,y2)
		#plt.show()

		
	else:
		y1=F(v)[1]
		y2=-2*np.cos(2*x)+10*np.sin(2*x)


	

	

	r=np.sqrt(np.sum(np.square(y1-y2))/len(x))
	z=np.array([dx,r,y1,y2])
	return z

N=np.arange(80,120,10)
DX=[]
R=[]
for j in range(len(N)):
	z=G(N[j])
	DX.append(z[0])
	R.append(z[1])


#plots
plt.subplot(1,2,2)
plt.plot(DX,R,label='error')
plt.xlabel('dx')
plt.ylabel('Error')


plt.subplot(1,2,1)
x=np.linspace(0,np.pi/4.0,N[-1])
dx=x[2]-x[1]
plt.plot(x,z[2],label="Numerical solution")
plt.plot(x,-2*np.cos(2*x)+10*np.sin(2*x),"*",label="Actual solution")
plt.xlabel("x")
plt.title('dx='+str(dx))
plt.ylabel("y=-2*np.cos(2*x)+10*np.sin(2*x)")
plt.legend()
plt.show()