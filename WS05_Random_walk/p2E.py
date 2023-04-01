#Note:- This and all my previous programs are coded in python 2.7.15
#       sorry for not writing this in earlier submissions.(Roll no.-17MS072)
import numpy as np 
import matplotlib.pyplot as plt 
#f'(x,t) wrt t
def f1(x1,x2,x3,t1):
        z=(x2+x3-2.0*x1)/(2.0*t1)
        return z

#initial condition function
def f2(a1,dx2,dt,t2):
        si=t2*dx2**2.0/dt
        a=np.exp(-a1**2/(2*si**2))/(2*np.pi*si**2)**0.5
        return a

N=10000#number of time-steps
M=1000#number of particles

x=np.arange(-2,2,0.01)#Xrange
#x=np.linspace(0,1,M)

t=np.linspace(0,1000,N)#time
px=np.zeros([len(x),N])#array for (x,t)
dt=t[2]-t[1]
dx=0.01
px[:,1]=f2(x,dx,dt,1)#input initial conditions

#Code for Euler
for i in range(1,len(t)-1):
        p1=px[1:len(px)-1,i]
        p2=px[2:,i]
        p3=px[0:len(px)-2,i]
        px[1:len(px)-1,i+1]=p1 + f1(p1,p2,p3,t[i])*dt

#Plots
plt.figure()

plt.plot(x,px[:,N-1],label='Time = '+str(t[N-1]))


plt.plot(x,px[:,15],label='Time = '+str(t[15]))


plt.plot(x,px[:,500],label='Time = '+str(t[500]))

plt.xlabel('x')
plt.ylabel('p(x,t)')
plt.title('p(x,t) v/s x')
plt.legend()

plt.show()
