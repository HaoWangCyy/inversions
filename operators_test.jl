include("operators.jl")
using Base.Test

function nodeAvg1D_test()

    # Tests the 1d Node averaging

    n = 10 # 10 cells, makes 11 nodes

    # define values on the nodes
    node_val = [i for i in 0:n]

    cell_avg_truth = [0.5 + i for i in 0:(n-1)]

    Av = nodeAvg(n)
    cell_avg_test = Av * node_val

    @test cell_avg_test == cell_avg_truth

end


function nodeAvg2D_test()

    nodes = [1 2 3 4; 5 6 7 8; 9 10 11 12]'
    cell_avg_truth = [(1+2+5+6)/4 (2+3+6+7)/4 (3 + 4 + 7 + 8)/4;
                      (5+6+9+10)/4 (6+7+10+11)/4 (7+8+11+12)/4]'

    Av = nodeAvg(3,2)

    cell_avg_test = Av*nodes[:]

    @test cell_avg_test == cell_avg_truth[:]
end 

function nodeAvg3D_test()

    # First test the averaging on a single cube
    cube = zeros(2,2,2)
    cube[:;1;1] = [1 2]
    cube[:;2;1] = [3 4]
    cube[:;1;2] = [5 6]
    cube[:;2;2] = [7 8]

    
    cell_avg_truth = mean(cube)
    Av = nodeAvg(1,1,1)
    cell_avg_test = Av*cube[:]
    @test cell_avg_test[1] == cell_avg_truth

    # Now test for dimensions
    cube = rand(4,3,5)
    Av = nodeAvg(3,2,4)
    
    @test size(Av) == (3*2*4, length(cube))

    # Do a numerical test on one cell
    av_cube = Av*cube[:]
    truth_cell = mean(cube[1:2;1:2;1:2])

    @test av_cube[1] == truth_cell
    
end
   

function nodeDiff1D_test()

    # Make  test vector
    d = [i for i in 0:9].^2
    truth = [d[i+1] - d[i] for i in 1:9]/2.

    dn = 2.0
    Av = nodeDiff(9, dn)

    test = Av*d
    @test test==truth 
end 
    
function nodeDiff2D_test()

    n1,n2 =  3,2
    dn1,dn2 = .1,.2
    # Make the test data
    nodes = [1,2,3,4,5,6,7,8,9,10,11,12]

    grad = zeros(n1*n2*2)

    grad = [ones(3*3)/dn1, ones(2*4)*4/dn2]

    G = nodeDiff(n1,n2, dn1, dn2)

    test_grad = G*nodes[:]

    @test test_grad == grad 
end 

function nodeDiff3D_test()

    # first test it on a single element
    cube = zeros(2,2,2)
    cube[:;1;1] = [1 2]
    cube[:;2;1] = [1 3]
    cube[:;1;2] = [1 4]
    cube[:;2;2] = [1 5]

    G = nodeDiff(1,1,1)

    truth_grad = [1, 2, 3, 4, 0, 1, 0, 1, 0, 2, 0, 2]
    
    grad = G * cube[:]

    @test truth_grad == grad
 
end 

function edgeAvg2D_test()

    n1,n2 = 2,2

    x_edge = [[1,2] [3,4] [5,6]]
    y_edge = [[7,8,9] [10,11,12]]
    edges = [x_edge[:], y_edge[:]]

    truth = [(1+3)/2 + (7+8)/2, (2+4)/2 + (8+9)/2, (3+5)/2 + (10+11)/2,
             (4+6)/2 + (11+12)/2]

    AvE = edgeAvg(n1,n2)

    test = AvE * edges

    @test_approx_eq test truth
    
  

end 


function edgeAvg3D_test()

    n1,n2, n3 = 1,1,1
    
    # Make an element, take the derivitive, then average the edge to test
    # dimensions
    cube = rand(n1+1,n2+1,n3+1)

    G = nodeDiff(n1,n2,n3)

    grad_cube = G*cube[:]

    AvE = edgeAvg(n1,n2,n3)
    avg_grad_cube = AvE * grad_cube

    @test_approx_eq avg_grad_cube mean(grad_cube[:])
end
    

function helmholtz1D_check(n)

    # Try a fictitious source test

    # Make the mesh size
    n_nodes = n + 1
    n_cells = n
    dx = 0.1

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
    RS = diagm(Av'*V)*q
    u_test = LS\RS

    # test that convergence is O(dx)
    @test_approx_eq_eps u_test u_truth dx
    return mean(abs(u_test-u_truth))
end

function helmholtz2D_check()

    # Geometry
    n1=100
    n2=100
    x = linspace(0,1,n1+1)
    y = linspace(0,1,n2+1)

    dx = x[2]-x[1]
    dy = y[2]-y[1]

    # Model
    w_sqr = 1.0
    m = ones(n1+1,n2+1).^2
    rho = ones(n1+1,n2+1)

    # truth data
    u_truth = zeros(n1+1,n2+1)
    for i in 1:n1+1
        for j in 1:n2+1
            u_truth[i,j] = cos(pi*x[i]) + cos(pi*y[j])
        end 
    end
    
    q = -(pi^2*rho.*u_truth) +(w_sqr*m.*u_truth)


    # Make all operators
    Av = nodeAvg(n1,n2)
    AvE = edgeAvg(n1,n2)
    G = nodeDiff(n1,n2)
    V = ones(n1*n2) * dx*dy
    
    # we are solving Au=q for u
    H = helmholtzNeumann(Av, AvE, G, V, Av*rho[:], w_sqr, Av*m[:])
    LS = H
    RS = diagm(Av'*V)*q[:]

    u_test = reshape(LS\RS, n1+1, n2+1)
    u_test_new = zeros(n1+1, n2+1)

    for i in 1:(n1+1)
        for j in 1:(n2+1)
        u_test_new[i,j] = u_test[end-(i-1), end-(j-1)]
        end
    end
    return
end

function helmholtz1D_converge()

    # Check the convergence
    nsteps = 5
    error = zeros(nsteps)
 

    cells = [2^i for i in 6:(6+nsteps-1)]

    for i in 1:nsteps

        error[i] = helmholtz1D_check(cells[i])
    end
    
    rate = error[1:end-1]./error[2:end]

    # Test that when halfing the grid size we converge to a true answer
    # in steps of O(h^2)
    @test_approx_eq_eps rate ones(nsteps-1)*4 0.5
end
    

function helmholtz2D_check()


    return
end 

function helmholtz3D()
    
    n1,n2, n3 = 10,12, 14

    w_sqr = 1
    m = ones(n1,n2,n3)
    rho = ones(n1,n2, n3)
    q = zeros(n1+1,n2+1, n3+1)
    q[n1/2, n2/2, n3/2] = 1.0
        
    # Make all operators
    Av = nodeAvg(n1,n2,n3)
    AvE = edgeAvg(n1,n2,n3)
    G = nodeDiff(n1,n2,n3)

    # we are solving Au=q for u
    A = G'*diagm(AvE'*rho[:])*G + w_sqr*diagm(Av'm[:])

    # solve it
    u = A\q[:]
    u = reshape(u,n1+1, n2+1, n3+1)
    return
end 
    
nodeAvg1D_test()
nodeAvg2D_test()
#nodeAvg3D_test()

edgeAvg2D_test()
#edgeAvg3D_test()

nodeDiff1D_test()
nodeDiff2D_test()
#nodeDiff3D_test()

helmholtz1D_converge()
#helmholtz2D()
#helmholtz3D()
