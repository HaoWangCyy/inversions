using PyCall
@pyimport matplotlib.pyplot as plt


include("operators.jl")

n1,n2, n3 = 10,12, 14

w_sqr = 1
m = ones(n1,n2,n3)
rho = ones(n1,n2, n3)
q = zeros(n1+1,n2+1, n3+1)
q[5;5; 1] = 1.0

# Make all operators
Av = nodeAvg(n1,n2,n3)
AvE = edgeAvg(n1,n2,n3)
G = nodeDiff(n1,n2,n3)

# we are solving Au=q for u
A = G'*diagm(AvE'*rho[:])*G + w_sqr*diagm(Av'm[:])

# solve it
u = A\q[:]
u = reshape(u,n1+1, n2+1, n3+1)


plt.imshow(u[:,:,7])
plt.show()
