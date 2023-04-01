import numpy as np
import matplotlib.pyplot as plt
def g(k,u):
        y=np.linspace(2,1000,u)

        t=np.linspace(0,20,u)
        dt=t[2]-t[1]

        x=np.linspace(k,1000,u)
        x[0]=1
        x[1]=np.cos((k**0.5)*t[1])

        for i in range(2,u):
                x[i]=2*x[i-1]-x[i-2]-k*(x[i-1])*dt**2
                y[i-1]=(x[i]-x[i-2])/(2*dt)

        y[0]=x[1]/2*dt

        z=y[1:-1]
        t1=t[1:-1]
        
        v=-(k**0.5)*np.sin((k**0.5)*t1)
        e=(sum(v-z)**2/len(v))**0.5
        
        return e

a=np.zeros(1500)
b=np.zeros(1500)

for j in range(1000,2500):        
        t=np.linspace(0,2,j)
        dt=t[2]-t[1]
        a[j-1000]=g(0.34,j)
        b[j-1000]=dt

plt.figure()
plt.subplot(1,2,1)
plt.plot(b,a)
plt.title('Error v/s dt')
plt.xlabel('dt')
plt.ylabel('Error')

plt.subplot(1,2,2)
plt.plot(b**2,a)
plt.title('Error v/s dt^2')
plt.xlabel('dt^2')
plt.ylabel('Error')


plt.show()


        
        
        
                
        
        

