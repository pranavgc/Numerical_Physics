#Coded in Python 3.7.3
import numpy as np
import numpy.linalg as la 

n=int(input("order of matrix = "))

a=np.zeros((n,n))
for i in range(n):
	a[i,:]=input('Enter '+str(i)+'th row = ').split(' ')
#For Q,R		
def gs(A):
	Q=np.zeros((n,n))
	Q[:,0]=A[:,0]/la.norm(A[:,0])
	
	for i in range(1,n):
		a=np.zeros(n)
		for j in range(i):
			a+=np.matmul(np.transpose(Q[:,j]),A[:,i])*Q[:,j]
		Q[:,i]=A[:,i]-a
		Q[:,i]=Q[:,i]/la.norm(Q[:,i])
	R=np.zeros((n,n))
	

	for i in range(n):
		for j in range(i,n):
			R[i,j]=np.dot(Q[:,i],A[:,j])
	
	return Q,R

#Checking Convergence of offdaigonal elements
of=1
while of>=0.002:
	A=np.matmul(gs(A)[1],gs(A)[0])
	V=[]
	
	for i in range(n):
		for j in range(n):
			if not i==j:
				V.append(A[i,j])
	
	of=max(V)
E=[]

for i in range(n):
		E.append(A[i,i])

print("eigenvalues =", E)
	
	