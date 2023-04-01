import numpy as np
import matplotlib.pyplot as plt
#Number of bins\
N=1000
#Number of random numbers\
n=10000


def g(u1,si1,N,n):
    x1=(np.random.rand(n))
    t1=(np.random.rand(n))

    #Function for generation\

    def f(u,si,x,t):
        return u+si*(np.cos(2*np.pi*t))*(-2*np.log(x))**0.5

    #Function for verification\
    def V(u,si,x):
        return np.exp((-(x-u)**2)/(2.0*si**2))/((2*np.pi*si**2)**0.5)

    #Code for sorting into bins\
    a = np.sort(f(u1,si1,x1,t1))

    b=(a+np.abs(np.amin(a)))/(np.amax(a)-np.amin(a))
    '''print a
    print b
    print np.amin(a),np.amax(a)-np.amin(a)'''

    da=(np.amax(a)-np.amin(a))/N
    y=(b*N)
    y=y.astype(int)
    c=[]
    z=list(y)
    for i in range(N):
            z1=z.count(i)
            c.append(z1)
            
    
    w1=np.linspace(np.min(a),np.max(a),len(a))
    g=np.asarray(c)

    #Code for normaliztion\
    g1=np.sum(g*da)
    w=np.arange(np.min(a),np.max(a),da)
    w=w[:len(g)]
    return w,g/g1,V(u1,si1,w)
    #print g1
    
    #plt.plot(w,g/g1)
    #plt.hist(a,N)
    #plt.show()
    #v=np.sum(g/g1)*da
    #print v

#Values of mean and stddev

u2=[0,2,2]
si2=[1,1,16]

#Plots
plt.figure(1)
for i in range(len(u2)):
    a1,b1,c1=g(u2[i],si2[i],N,n)
    plt.subplot(2,2,i+1)
      
    plt.plot(a1,b1,label='Generated Gaussian')
    plt.plot(a1,c1,label='Formulated Gaussian')
    plt.legend()
    plt.title('Mean = '+str(u2[i])+',STDDEV='+str(si2[i]))
    plt.xlabel('X')
    plt.ylabel('P(X)')
plt.show()
