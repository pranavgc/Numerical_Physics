import numpy as np
import matplotlib.pyplot as plt

#plt.show(block=True) close figure to continue code.
N=np.array([100,200,400])


for n in N:

        K = np.linspace(0,6,50)
        t = np.linspace(0,10,1000)
        

        x1 = np.zeros([n,len(t)]) 

        R = np.zeros(len(K)) 



        #omegas for n oscillators
        w = np.random.standard_cauchy(n)

        #initial condition for oscillators
        x1[:,0] = np.random.uniform(0,2*np.pi,n)
        h = t[3]-t[2]


        def r(x):
            r1 = np.absolute(np.sum(np.exp(x*1j)))/n
            w1 = np.angle(np.sum(np.exp(x*1j)))/n
            return r1,w1




        def f1(y):
                a,b=r(y)
                return w + (k*a)*np.sin(b-y)
             
            

        for j in range(len(K)):
            k = K[j]
            for i in range(len(t)-1):
                x = x1[:,i]
                k1 = h*f1(x)
                k2 = h*f1(x+k1/2.0)
                k3 = h*f1(x+k2/2.0)
                k4 = h*f1(x+k3)
                t[i+1] = t[i] + h
                x1[:,i+1] = x + ((1.0/6)*(k1+2*k2+2*k3+k4))
            
            a1,b1=r(x1[:,len(t)-30])
            R[j]=a1
            
        
        plt.scatter(K,R)
        plt.title('r v/s K for N='+str(n))
        plt.xlabel('K')
        plt.ylabel('|r|')
        plt.show()
