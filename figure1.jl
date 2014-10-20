using PyCall
@pyimport matplotlib.pyplot as plt

include("operators_test.jl")

# Make the mesh size
n = 1024
n_nodes = n + 1
n_cells = n

rho = ones(n_nodes)
x = linspace(0,1,n_nodes)
dx = x[2]-x[1]
w_sqr = 0.011
m = randn(n_nodes).^2.0
q = (rho.*((4*(pi^2)*cos(2*pi*x)) + (16*(pi^2)*cos(4*pi*x))) +
     (w_sqr*m.*(cos(2*pi*x) + cos(4*pi*x))))
     
# Truth with 0 as boundary conditions(dirichlet)
u_truth =cos(2*pi*x) + cos(4*pi*x)

# Make all operators
Av = nodeAvg(n_cells)
AvE = edgeAvg(n_cells)
G = nodeDiff(n_cells, dx)
V = ones(n_cells) * dx

# move to cell centers
m_cell = Av*m
rho_cell = Av*rho
    
# we are solving Au=q for u

LS = helmholtzNeumann(Av, AvE, G,V,rho_cell, w_sqr, m_cell)
RS = spdiagm(Av'*V)*q
u_test = LS\RS

plt.figure()
plt.subplot(211)
plt.plot(x,u_truth, color="blue")
plt.ylim((-3,3))
plt.ylabel("U(x) analytic")
plt.subplot(212)
plt.plot(x, u_test, color="red")
plt.ylim((-3,3))
plt.xlabel("x [arbitrary units]")
plt.ylabel("U[x] solution")
#plt.subplot(313)
#plt.plot(x, u_truth-u_test)
#plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
#plt.title("residual error")
#plt.xlabel("x [arbitrary units]")
#plt.ylabel("error [arbitrary units]")
#plt.tight_layout()

plt.savefig("figure1.eps")
# Check the convergence
nsteps = 10
error = zeros(nsteps)
 

cells = [2^i for i in 1:(nsteps)]

for i in 1:nsteps

    error[i] = helmholtz1D_check(cells[i])
end
    
rate = error[1:end-1]./error[2:end]

plt.figure()
plt.plot(cells[2:end],rate)
plt.ylim((0,10))
plt.xscale("log")
plt.ylabel("convergence rate")
plt.xlabel("# of nodes")
plt.savefig("figure2.eps")






