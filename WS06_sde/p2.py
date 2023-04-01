#Coded in Python 2.7.15
import numpy as np
import matplotlib.pyplot as plt
import math

def f1(n1,t1,z1,k11,k21):#dn
	 	return k11*t1-k21*n1*t1+z1*(t1*(k11+k21*n))**0.5

def f0(t0,k10,k20):#theoretical n(t)
	return k10*(1-np.exp(-k20*t0))/float(k20)


#def dp(p,p1,p2,n,n2,k1,k2):
    #return k2*n2*p1+k1*p2-n*k2*p-k1*p

def p(n,k1,k2):#theoretical P(n) in steady state
    return (k1/k2)**n*np.exp(-k1/k2)/math.factorial(n)

n=100#Number of realizations

k1=1.5
k2=0.03

t=np.linspace(0,250,1000)#time
dt=t[2]-t[1]

z=np.zeros([n,len(t)])#for storing random numbers
for j in range(n):
    z[j,:]=np.random.randn(len(t))

N=np.zeros([n,len(t)])#for storing simulated n(t)
y=np.zeros(len(t))#for storing theoretical n(t)

for i in range(len(t)-1):
    
    N[:,i+1]=N[:,i]+f1(N[:,i],dt,z[:,i+1],k1,k2)
    y[i+1]=f0(t[i+1],k1,k2)


a=np.average(N,axis=0)#simulated n(t)
#print a
#print y

#Plots for n v/s t

plt.subplot(1,2,1)
plt.plot(t,a,label="mean trajectory")
plt.plot(t,y,label="formulated trajectory")
plt.xlabel('time')
plt.ylabel('Concentration of mrna(n)')
plt.title('Conc of mrna v/s time')
plt.legend()
plt.show(block=False)

#Steady state reaches by t=100

v=N[:,len(t)-1].astype(int)
 
print "mean=",np.mean(N[:,len(t)-1])
print "expected mean=",k1/k2


#Plots for P(n) v/s n
plt.subplot(1,2,2)
p2=np.vectorize(p)
plt.hist(v,45,density=True,label='simulated result')
plt.plot(v,p2(v,k1,k2),'*',label='expected result')
plt.xlabel('Concentration of mrna(n)')
plt.ylabel('P(n)')
plt.title('P(n) v/s n')
plt.legend()
plt.show()



