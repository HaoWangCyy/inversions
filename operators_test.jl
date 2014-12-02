include("operators.jl")
include("solvers.jl")

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
    return
end 

function nodeDiff3D_test()

    # first test it a 2X2 cube
    cube = [i for i in 1:27]
    dx,dy,dz = .1,.2,.3
    G = nodeDiff(2,2,2, dx,dy,dz)

    
    truth_grad = [ones(2*3*3)/dx, 3*ones(2*3*3)/dy, 9*ones(2*3*3)/dz]
    
    grad = G * cube

    @test_approx_eq truth_grad  grad
    return
end 

function edgeAvg2D_test()

    n1,n2 = 2,2

    x_edge = [[1,2] [3,4] [5,6]]
    y_edge = [[7,8,9] [10,11,12]]
    edges = [x_edge[:], y_edge[:]]



    AvE = edgeAvg(n1,n2)

    test = AvE * edges

     truth = [(1+3)/2 + (7+8)/2, (2+4)/2 + (8+9)/2, (3+5)/2 + (10+11)/2,
              (4+6)/2 + (11+12)/2]
    
    @test_approx_eq test truth
    return
end 


function edgeAvg3D_test()

    # first test it a 2X2 cube
    cube = [i for i in 1:27]
    dx,dy,dz = .1,.2,.3
    
    G = nodeDiff(2,2,2,dx,dy,dz)
    grad = G * cube

    AvE = edgeAvg(2,2,2)
    
    truth = ones(2*2*2) * (1/dx) + 3*ones(2*2*2)*(1/dy) + 9*ones(2*2*2)*(1/dz)
    test = AvE * grad

    @test_approx_eq test truth
    
    
end
    

function helmholtz1D_check(n)

    # Try a fictitious source test

    # Make the mesh size
    n_nodes = n + 1
    n_cells = n

    rho = ones(n_nodes)
    x = linspace(0,1,n_nodes)
    dx = x[2]-x[1]
    w = 0.2
    w_sqr = w^2
    m = randn(n_nodes).^2.0
    q = -(-rho.*((4*(pi^2)*cos(2*pi*x)) + (16*(pi^2)*cos(4*pi*x))) +
         (w_sqr*m.*(cos(2*pi*x) + cos(4*pi*x))))
    
    # Truth with 0 as boundary conditions(dirichlet)
    u_truth = cos(2*pi*x) + cos(4*pi*x)
    
    # put scalars onto cell centers
    Av = nodeAvg(n_cells)
    rho = Av*rho
    m = Av*m
    
    u_test = helmholtzNeumann(rho, w, m, q, dx)

    # test that convergence is O(dx)
    #@test_approx_eq_eps u_test u_truth dx
    return mean(abs(u_test-u_truth))
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
    @test_approx_eq_eps rate[end] 4 0.3
end
    

function helmholtz2D_check(n1,n2)
    # Geometry
    x = linspace(0,1,n1+1)
    y = linspace(0,1,n2+1)

    d1 = x[2]-x[1]
    d2 = y[2]-y[1]
    dv=(d1,d2)
    # Model
    w = 2.0
    w_sqr = w^2
    
    m = randn(n1+1,n2+1).^2 
    rho = ones(n1+1,n2+1).^2

    # truth data
    u_truth = zeros(n1+1,n2+1)
    for i in 1:n1+1
        for j in 1:n2+1
            u_truth[i,j] = cos(pi*x[i]) + cos(pi*y[j])
        end 
    end
    
    q = (pi^2)*rho.*u_truth - (w_sqr*m.*u_truth)


    # put scalars on centers
    Av = nodeAvg(n1,n2)
    m = reshape(Av*m[:], n1,n2)
    rho = reshape(Av*rho[:], n1,n2)

    H, Q = helmholtz(rho, w, m, dv)
   
    
    u_test = helmholtzNeumann(rho, w, m, q, (d1,d2))

    #print( maximum(H*u_truth[:]-Q*q[:]))
    #@test_approx_eq_eps u_test u_truth[:] d1*d2
    return mean(abs(H*u_truth[:]-Q*q[:]))
    
end

function helmholtz2D_converge()
    n = [4,8,16,32,64, 128, 256, 512, 1024]

    error = zeros(9)
    
    for i in 1:9
        error[i] = helmholtz2D_check(n[i],n[i])
    end

    rate = error[1:end-1]./ error[2:end]

    return rate
    #@test_approx_eq_eps rate[end] 4.0 .3
end

function helmholtz3D_converge()
    n = [4,8,16,32,64]

    error = zeros(5)
    
    for i in 1:5
        error[i] = helmholtz3D_check(n[i],n[i], n[i])
        
    end

    rate = error[1:end-1]./ error[2:end]

    @test_approx_eq_eps rate[end] 4.0 .3
end
    
function helmholtz3D_check(n1,n2,n3)

    x = linspace(0,1,n1+1)
    y = linspace(0,1,n2+1)
    z = linspace(0,1,n3+1)

    d1 = x[2]-x[1]
    d2 = y[2]-y[1]
    d3 = z[2]-z[1]

    dv = (d1,d2,d3)
    # Model
    w = 1.0
    w_sqr = w^2
    m = randn((n1+1),(n2+1),(n3+1)).^2
    rho = ones((n1+1),(n2+1),(n3+1)).^2

    # truth data
    u_truth = zeros(n1+1,n2+1, n3+1)
    for i in 1:n1+1
        for j in 1:n2+1
            for k in 1:n3+1
                u_truth[i,j,k] = cos(pi*x[i]) + cos(pi*y[j]) + cos(pi*z[k])
            end 
        end
    end 
 
    q = (pi^2*rho.*u_truth) -(w_sqr*m.*u_truth)
    
    # put scalars on centers
    Av = nodeAvg(n1,n2,n3)
    m = reshape(Av*m[:], n1,n2,n3)
    rho = reshape(Av*rho[:], n1,n2,n3)

    # we are solving Au=q for u
    u_test = helmholtzNeumann(rho, w, m, q, (d1,d2,d3))


    return mean(abs(u_test-u_truth))
end 
    


function helmholtzSensTest()

    n = 50
    P = speye(n+1)

    w = 10.0

    m = 1./(randn(n)*2500).^2

    rho = randn(n).^2 * 2500

    q = zeros(n+1)
    q[20] = 1.0

    dv = 1.0

    sens = helmholtzSensitivity(P, rho, w, m, q, dv)

end 

    
function derivativeTest1D()

    n = 100

    q = randn(n+1).^2
    m = randn(n).^2
    w = 10.0
    rho = ones(n)
    dv = 10.0
    v = rand(size(m))
    
    U = helmholtzNeumann(rho, w, m,q,dv)

    G = helmholtzDerivative(U,w,dv)

    diff_lin = zeros(10)
    diff_quad = zeros(10)
    h_ = zeros(10)

    P = speye(n+1)
    J = helmholtzSensitivity(P, rho, w, m, q, dv)
    
    for i in 1:10

        h = 10.0^(-i)

        Ui = helmholtzNeumann(rho, w, m+(h*v),q,dv)
        
        diff1 = norm(Ui-U)
        diff2 = norm(Ui - U - (J*h*v))
        
        diff_lin[i] = diff1
        diff_quad[i] = diff2
        h_[i] = h
        
    end 

    lin_rate = diff_lin[2:end] ./ diff_lin[1:end-1]
    quad_rate = diff_quad[2:end] ./ diff_quad[1:end-1]

    @test_approx_eq_eps lin_rate[5] 1/10. .01
    @test_approx_eq_eps quad_rate[5] 1/100. .001
end

function derivativeTest2D()

    n1,n2 = 15,15
    
    q = randn(n1+1, n2+1).^2
    m = randn(n1,n2).^2
    w = 10.0
    rho = ones(n1,n2)
    dv = (10.0,10)
    v = rand(size(m))
    
    U = helmholtzNeumann(rho, w, m,q,dv)

    G = helmholtzDerivative(U,w,dv)

    diff_lin = zeros(10)
    diff_quad = zeros(10)
    h_ = zeros(10)

    P = speye((n1+1)*(n2+1))
    J = helmholtzSensitivity(P, rho, w, m, q, dv)
    
    for i in 1:10

        h = 10.0^(-i)

        Ui = helmholtzNeumann(rho, w, m+(h*v),q,dv)
        
        diff1 = norm(Ui[:]-U[:])
        diff2 = norm(Ui[:] - U[:] - (J*h*v[:]))
        
        diff_lin[i] = diff1
        diff_quad[i] = diff2
        h_[i] = h
        
    end 

    lin_rate = diff_lin[2:end] ./ diff_lin[1:end-1]
    quad_rate = diff_quad[2:end] ./ diff_quad[1:end-1]

    @test_approx_eq_eps lin_rate[5] 1/10. .01
    @test_approx_eq_eps quad_rate[5] 1/100. .001
end

function adjointTest()

    n = 1000

    q = linspace(0,10,n+1).^2
    m = linspace(10,20,n).^2
    w = 10.0
    rho = ones(n)
    dv = 10.0
    v = ones(size(m))*.01

    P = speye(n+1)
    J = helmholtzSensitivity(P, rho, w, m, q, dv)

    y = linspace(20,50,n+1)
    x = linspace(0,10,n)

    adj_test = ((J*x)'*y) - (x'*(J'*y))
    @test_approx_eq_eps  adj_test 0.0 1e-20

end


function adjointTest2D()

    n1,n2 = 25,25
    
    q = zeros(n1+1, n2+1)
    q[[2,4,6],10] = 100
    m = reshape(linspace(1,2000, n1*n2), n1,n2)
    w = 2.0
    rho = ones(n1,n2)
    dv = (10.0,10.0)
    
    P = speye((n1+1)*(n2+1))
    J = helmholtzSensitivity(P, rho, w, m, q, dv)

    y = linspace(0,2,(n1+1)*(n2+1))
    x = linspace(0,10,(n1)*(n2))

    adj_test = ((J*x)'*y) - (x'*(J'*y))
    @test_approx_eq_eps  adj_test 0.0 1e-16

end


function jacobianConvergence(rho, w, m_pad, m, Q, P, A,dv,S,s,Ia)
    
    v = rand(size(m_pad))

    U = helmholtzNeumann(rho, w, m_pad,Q,dv, S,s)
    D = P'*reshape(U, prod(size(U)[1:2]), size(P)[2])

    diff_lin = zeros(10)
    diff_quad = zeros(10)
    
    for i =1:10
        h = 10.0^(-i)

        dm = v*h

        
        print(size(Ia'*dm[:]))
        
        J = jacobianv(U, A, P, w, dv, Ia'*dm[:], s, Ia)
        Ui = helmholtzNeumann(rho, w, m_pad + dm, Q, dv, S,s)
        
        Di = P'*reshape(Ui, prod(size(U)[1:2]), size(P)[2])
        
        diff_lin[i] = norm(Di[:]-D[:])
     
        diff_quad[i] = norm(Di[:] - D[:] - J[:])
    end 
                         
        
   return diff_lin, diff_quad     
    
end 

    
    
function adjointVectorTest(u,m, A, P, w, dv, s, Ia)

    

    v = randn(size(m))*1e-4
    V = randn(size(P)[2], size(P)[2])*1e-4

    JTV = jacobianTw(u, A, P, w, dv, V, s, Ia)
    Jv = jacobianv(u, A, P, w, dv, v, s, Ia)


    adj_test = JTV'*v[:] - V[:]'*Jv[:]

    print(adj_test)
    @test_approx_eq_eps  adj_test 0.0 1e-20
end 
