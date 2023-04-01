import numpy as np
import matplotlib.pyplot as plt
#Number of bins\
N=1000
#Number of random numbers\
n=10000


def g(u1,N,n):
    x1=(np.random.rand(n))
    for i in range(len(x1)):
        x1[i]=0.01+(0.99-0.01)*x1[i]
    
    #Function for generation\

    def f(u,x):
        return u*np.tan(np.pi*(x-0.5))

    #Function for verification\
    def V(u,x):
        return u/((u**2+x**2)*np.pi)

    #Code for sorting into bins\
    a = np.sort(f(u1,x1))

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
    return w,g/g1,V(u1,w),a

    #print g1    
    #plt.plot(w,g/g1)
    #plt.hist(a,N)
    #plt.show()
    #v=np.sum(g/g1)*da
    #print v


#Values of gamma

u2=[2,3,6]
#si2=[1,1,16]

#Plots
plt.figure(1)
for i in range(len(u2)):
    a1,b1,c1,d=g(u2[i],N,n)
    plt.subplot(2,2,i+1)
      
    plt.plot(a1,b1,label='Generated Lorentzian')
    plt.plot(a1,c1,label='Formulated Lorentzian')
    #plt.hist(d,1000)
    plt.legend()
    plt.title('Gamma = '+str(u2[i]))
    plt.xlabel('X')
    
    plt.ylabel('P(X)')
    
plt.show()
