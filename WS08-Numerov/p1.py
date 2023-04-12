
#Coded in Python 3.7.4
import numpy as np 
import matplotlib.pyplot as plt 
import sys


def e(E,n1):
	#Potential
	def V():
		return 0.0
	def phi(E,d):
		return 1.0+2*(E-V())*(d**2.0)/12.0


	x=np.linspace(-5,5,n1)
	dx=x[1]-x[0]
	
	psi=np.zeros(len(x))

	psi[1]=0.00001
	#Single sided Shooting
	for i in range(1,len(x)-1):
		psi[i+1]=(psi[i]*(12-10*phi(E,dx))-psi[i-1]*phi(E,dx))/float(phi(E,dx))
	return psi

#Energy
E=np.linspace(0,0.1,100)
n=100
x=np.linspace(-5,5,n)
for i in range(len(E)):
	a=e(E[i],n)
	if np.all(a>=0.0)==True and np.abs(a[-1]-0)<0.00001:
		psi=a
		

		print('Ground State Energy ',E[i])

		print('Error ',np.abs(a[-1]-0))

		plt.title('Ground State Energy '+str(E[i]))
		
		plt.plot(x,psi,label='psi')
		plt.xlabel('x')
		plt.ylabel('psi(x)')
		plt.legend()
		
		plt.xlim(-6,6)
		plt.ylim(0,np.max(a)+0.0002)
		plt.show()
		sys.exit()

	
