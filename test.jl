using PyCall
@pyimport matplotlib.pyplot as plt


include("operators.jl")

# Geometry
n1=10
n2=10
x = linspace(0,1,n1+1)
y = linspace(0,1,n2+1)

dx = x[2]-x[1]
dy = y[2]-y[1]

# Model
w_sqr = 1.0
m = randn((n1+1)*(n2+1)).^2
rho = ones((n1+1)*(n2+1)).^2

# truth data
u_truth = zeros(n1+1,n2+1)
for i in 1:n1+1
    for j in 1:n2+1
        u_truth[i,j] = cos(pi*x[i]) + cos(pi*y[j])
    end 
 end
 
q = -(pi^2*rho.*u_truth[:]) +(w_sqr*m.*u_truth[:])


# Make all operators
Av = nodeAvg(n1,n2)
AvE = edgeAvg(n1,n2)/2.0
G = nodeDiff(n1,n2)/dx
V = ones(n1*n2) * dx*dy

# we are solving Au=q for u
H = helmholtzNeumann(Av, AvE, G, V, Av*rho[:], w_sqr, Av*m[:])
LS = H
RS = diagm(Av'*V)*q

u_test = reshape(LS\RS, n1+1, n2+1)
u_test_new = zeros(n1+1, n2+1)

for i in 1:(n1+1)
    for j in 1:(n2+1)
        u_test_new[i,j] = u_test[end-(i-1), end-(j-1)]
    end
end

plt.figure()
plt.subplot(121)
plt.imshow(u_test_new)
plt.subplot(122)
plt.imshow(u_truth)
plt.show()

plt.figure()
plt.subplot(121)
plt.plot(u_test_new[:])
plt.subplot(122)
plt.plot(u_truth[:])
plt.show()


