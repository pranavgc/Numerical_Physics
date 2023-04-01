import matplotlib.pyplot as plt
import numpy as np
w=3
w2=np.linspace(0.5,10,100)
c=np.arange(0.1,0.7,0.1)
for l in c:
	A=np.zeros(len(w2))	
	for j in range(len(w2)):
		

		def f1(x2):
			return x2


		def f2(x1,x2,l,t):
			return -2*l*w*x2-w**2*x1+F*np.sin(w1*t)
		w1=w2[j]
		F=10
		t=np.linspace(0,300,1000)
		dt=t[2]-t[1]
		x1=np.zeros(len(t))
		x2=np.ones(len(t))
		x1[0]=0
		x2[0]=4
		for i in range(len(t)-1):
			k11=f1(x2[i])
			k12=f2(x1[i],x2[i],l,t[i])
			
			k21=f1(x2[i]+dt/2*k12)
			k22=f2(x1[i]+dt/2*k11,x2[i]+dt/2*k12,l,t[i]+dt/2)
			
			k31=f1(x2[i]+dt/2*k22)
			k32=f2(x1[i]+dt/2*k21,x2[i]+dt/2*k22,l,t[i]+dt/2)
			
			k41=f1(x2[i]+dt*k32)
			k42=f2(x1[i]+dt*k31,x2[i]+dt*k32,l,t[i]+dt)
			
			x1[i+1]=x1[i]+dt/6*(k11+2*k21+2*k31+k41)
			x2[i+1]=x2[i]+dt/6*(k12+2*k22+2*k32+k42)
		
		A[j]=np.amax(x1[500::])
		n=np.where(A==np.amax(A))
	print('Resonance frequency',(w2/w)[n]*w,'for lambda',l)
		
	plt.plot(w2/w,A,label='lambda= '+str(l))
	plt.legend()
	plt.xlabel('w/w0')
	plt.ylabel('A(w)')
	plt.grid()
plt.show()



