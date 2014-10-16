using PyCall
@pyimport matplotlib.pyplot as plt

include("operators_test.jl")

n1 = 25
n2 = 25
n3 = 25
# Geometry
 x = linspace(0,1,n1+1)
 y = linspace(0,1,n2+1)
 z = linspace(0,1,n3+1)
 
 d1 = x[2]-x[1]
 d2 = y[2]-y[1]
 d3 = z[2]-z[1]

# Model
w_sqr = 1.0
m = linspace(1,100,(n1+1)*(n2+1)*(n3+1)).^2
rho = ones((n1+1)*(n2+1)*(n3+1)).^2

# truth data
u_truth = zeros(n1+1,n2+1, n3+1)
for i in 1:n1+1
    for j in 1:n2+1
        for k in 1:n3+1
            u_truth[i,j,k] = cos(pi*x[i]) + cos(pi*y[j]) + cos(pi*z[k])
        end 
    end
end 
    
q = (pi^2*rho.*u_truth[:]) +(w_sqr*m.*u_truth[:])
   
# Make all operators
Av = nodeAvg(n1,n2,n3)
AvE = edgeAvg(n1,n2,n3)
G = nodeDiff(n1,n2,n3,
             d1,d2,d3)
    
V = ones(n1*n2*n3) * d1*d2*d3

# we are solving Au=q for u
H = helmholtzNeumann(Av, AvE, G, V, Av*rho[:], w_sqr, Av*m[:])
LS = H
RS = spdiagm(Av'*V)*q

u_test = reshape(LS\RS, n1+1, n2+1, n3+1)


plt.figure()
plt.subplot(121)
plt.imshow(u_truth[:,:,10], extent=[0,1,0,1])
plt.title("U(x,y,10) analytic")
plt.subplot(122)
plt.imshow(u_test[:,:,10], extent = [0,1,0,1])

plt.title("U[x,y,10] solution")

plt.savefig("figure5.png")
plt.show()
# Check the convergence
cells =[1,2,4,8,16,32,64]
nsteps = length(cells)
error = zeros(nsteps)
 

cells =[1,2,4,8,16,32,64]

for i in 1:nsteps

    error[i] = helmholtz3D_check(cells[i], cells[i], cells[i])
end
    
rate = error[1:end-1] ./ error[2:end]

plt.figure()
plt.plot(cells[2:end].^3,rate)
#plt.yscale("log")
plt.xscale("log")
plt.ylabel("convergence rate")
plt.xlabel("# of nodes")
plt.savefig("figure6.png")

plt.show()





