using PyCall
@pyimport matplotlib.pyplot as plt


include("operators.jl")

n_nodes = 1000
n_cells = n_nodes-1
    # sim
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
    G = nodeDiff(n_cells)
    V = ones(n_cells) * dx^2
   
    
    
    m_cell = Av*m 
    rho_cell = Av*rho
    
    # we are solving Au=q for u
    A = G'*diagm(rho_cell)*G + diagm(Av'*(w_sqr*V.*m_cell))
    LS = A
    RS = diagm(Av'*V)*q
    u_test = LS\RS

plt.subplot(131)
plt.plot(u_truth)
plt.subplot(132)
plt.plot(u_test)
plt.subplot(133)
plt.plot(u_test-u_truth)
plt.show()


