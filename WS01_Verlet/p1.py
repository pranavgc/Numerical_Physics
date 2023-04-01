import numpy as np
import matplotlib.pyplot as plt
def g(b,n,k):
	t=np.linspace(0,10,1000)
	dt=t[2]-t[1]
	m=0.5*(b**2+k*n**2)

	y=np.linspace(b,1000,1000)
	x=np.linspace(n,1000,1000)
	E=np.linspace(m,1000,1000)
	

	y2=np.linspace(b,1000,1000)
	x2=np.linspace(n,1000,1000)
	E2=np.linspace(m,1000,1000)
	
	for i in range(1,1000):
		y[i]=y[i-1]-k*x[i-1]*dt
		x[i]=x[i-1]+y[i]*dt
		E[i]=0.5*(y[i]**2+k*x[i]**2)

		y2[i]=y2[i-1]-k*x2[i-1]*dt
		x2[i]=x2[i-1]+y2[i-1]*dt
		E2[i]=0.5*(y2[i]**2+k*x2[i]**2)
		
		
	#print y
	
	
	
		
	
	
	
	plt.figure().suptitle("Initial conditions theta="+str(n)+" theta.dot="+str(b)+" energy="+str(m))
	plt.subplot(1,2,2)
	plt.plot(t,E)
	plt.title("Euler Cromer Method")
	plt.legend()
	plt.xlabel('Time')
	plt.ylabel('Energy')

	plt.subplot(1,2,1)
	plt.plot(t,E2)
	plt.title("Euler Method")
	plt.legend()
	plt.xlabel("Time")
	plt.ylabel("Energy")
	
	plt.show()


#g(2,1,0.2)
#g(3,2,0.2)
g(5,2,0.2)


