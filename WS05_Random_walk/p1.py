#Note:- This and all my previous programs are coded in python 2.7.15
#       sorry for not writing this in earlier submissions.(Roll no.-17MS072)
import numpy as np
import matplotlib.pyplot as plt
import random as r
#Function for getting end position after completing walk
def f(N):
	
	
	z=0
	for i in range(N):
		y=r.choice([-1,1])
		z=z+y
		
	return z
n=100#Number of walkers
x=np.zeros(n)#array for storing end pos

n1=np.arange(100,1000,1)#Number of timesteps

x1=np.linspace(100,1000,len(n1))#array for storing var(x)

for k in range(len(n1)):
	for j in range(len(x)):
		x[j]=f(n1[k])


	z1=np.mean(x)
	x1[k]=np.var(x)
#Plots
plt.figure()
plt.subplot(1,2,2)
plt.plot(n1,x1)
plt.xlabel('N')
plt.ylabel('Var(x)')
plt.title('var(x) v/s N')

plt.subplot(1,2,1)
plt.hist(x,100,density=True)
plt.xlabel(' End position')
plt.ylabel('Number of Walkers')
plt.title('Number of Walkers v/s End position')
plt.show()
