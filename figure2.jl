using PyCall
@pyimport matplotlib.pyplot as plt

include("operators_test.jl")

n1 = 50
n2 = 60
# Geometry
x = linspace(0,1,n1+1)
y = linspace(0,1,n2+1)

d1 = x[2]-x[1]
d2 = y[2]-y[1]

# Model
w_sqr = 1.0
m = linspace(1,50, (n1+1)*(n2+1))
rho = ones((n1+1)*(n2+1)).^2

# truth data
u_truth = zeros(n1+1,n2+1)
for i in 1:n1+1
    for j in 1:n2+1
        u_truth[i,j] = cos(pi*x[i]) + cos(pi*y[j])
    end 
end
    
q = (pi^2*rho.*u_truth[:]) +(w_sqr*m.*u_truth[:])
    
    
# Make all operators
Av = nodeAvg(n1,n2)
AvE = edgeAvg(n1,n2)
G = nodeDiff(n1,n2, d1, d2)
V = ones(n1*n2) * d1*d2

# we are solving Au=q for u
H = helmholtzNeumann(Av, AvE, G, V, Av*rho[:], w_sqr, Av*m[:])
LS = H
RS = spdiagm(Av'*V)*q

u_test = reshape(LS\RS, n1+1, n2+1)


plt.figure()
plt.subplot(121)
plt.imshow(u_truth, extent=[0,1,0,1])
plt.title("U(x,y) analytic")
plt.subplot(122)
plt.imshow(u_test, extent = [0,1,0,1])

plt.title("U[x,y] solution")

plt.savefig("figure3.png")
plt.show()
# Check the convergence
nsteps = 10
error = zeros(nsteps)
 

cells = [2^i for i in 1:(nsteps)]

for i in 1:nsteps

    error[i] = helmholtz2D_check(cells[i], cells[i])
end
    
rate = error[1:end-1]./error[2:end]

plt.figure()
plt.plot(cells[2:end].^2,rate)
#plt.yscale("log")
plt.xscale("log")
plt.ylabel("convergence rate")
plt.xlabel("# of nodes")
plt.savefig("figure4.png")

plt.show()





