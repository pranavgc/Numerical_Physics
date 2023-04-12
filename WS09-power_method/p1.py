
#Coded in Python 3.7.3
import numpy as np 
import numpy.linalg as la 
#Input matrix
n=int(input("order of matrix = "))

a=np.zeros((n,n))
for i in range(n):
	a[i,:]=input('Enter '+str(i)+'th row = ').split(' ')


#power method getting largest eigenvalue
def D(a):
	
	def A(a1,a2,y,d):
		z=np.matmul(a1,a2)
		z2=np.matmul(a1,a2)/np.max(np.matmul(a1,a2))
		z3=np.matmul(np.transpose(z),y)

		return z3/d,z2


	#a=np.array(([1,3],[2,2]))
	#a2=np.array(([1,3],[2,2]))
	f1=9
	f2=5
	s=1

	x=np.array(([1],[3]))
	a2=np.matmul(a,x)
	y=np.array(([3],[1]))

	while np.abs(f2-f1)>1e-09:
		f1=f2
		f2,s1=A(a,a2,y,s)
		#print(s1)
		a2=s1
		s=np.matmul(np.transpose(a2),y)

	else:
		return f2,a2					


#getting other eigenvalues
e=np.zeros(n)
f=np.zeros([n,n])
e[0],f[:,[0]]=D(a)	
for i in range(1,n):
	e[i],f[:,[i]]=D(a-e[i-1]*np.matmul(f[:,[i-1]]/la.norm(f[:,[i-1]]),np.transpose(f[:,[i-1]]/la.norm(f[:,[i-1]]))))
print('eigen Values = 'e)
print('eigenvectors='f)

