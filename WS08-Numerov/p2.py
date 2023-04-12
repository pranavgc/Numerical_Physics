
#Coded in Python 3.7.4
import numpy as np
import matplotlib.pyplot as plt
import sys



def e(E,n1):
	#Potential
	def V(x2):
		if np.abs(x2)<=5.0:
			return -2.0
		else:
			return 0.0
	V=np.vectorize(V)
	
	x=np.linspace(-10,10,n1)
	dx=x[1]-x[0]
	dx1=-dx
	
	
	#phi
	k=2*(E-V(x))
	phi=1+k*(dx**2)/12.0
	psi=np.zeros(int(len(x)/2))
	psi[1]=1e-05
	psi1=np.zeros(int(len(x)/2))
	psi1[1]=1e-05

	#Double Shooting
	for i in range(2,int(len(x)/2)):
		psi[i]=(psi[i-1]*(12-10*phi[i-1])-psi[i-2]*phi[i-2])/float(phi[i])
		psi1[i]=(psi1[i-1]*(12-10*phi[i-1])-psi1[i-2]*phi[i-2])/float(phi[i])
	
	#Gamma
	g=(psi[int(len(x)/2)-1]-psi[int(len(x)/2)-2])/(2*dx*psi[int(len(x)/2)-1])
	g1=(psi1[int(len(x)/2)-1]-psi1[int(len(x)/2)-2])/(2*dx1*psi1[int(len(x)/2)-1])
	

	z=[g-g1,psi,psi1]
	return z

E=np.linspace(-2,0,100)
n=1000
x=np.linspace(-10,10,n)

for i in range(len(E)):
	a=e(E[i],n)
	
	z2=a[0]
	z=a[1]
	c=a[2]
	
	
	if np.all(z>=0.0)==True and np.all(c>=0.0)==True  and np.abs(z2)<0.01: 
		
		psi=np.concatenate((z,c[::-1]))

		psi=psi/np.sum(psi**2)

		print('Ground State Energy ',E[i])

		print('Error ',np.abs(z2))
		plt.title('Ground State Energy '+str(E[i]))
		plt.plot(x,psi,label='Normalized psi')
		plt.xlabel('x')
		plt.ylabel('psi(x)')
		plt.legend()
		
		plt.show()
		sys.exit()
