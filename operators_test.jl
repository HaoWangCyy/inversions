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
    truth = [d[i+1] - d[i] for i in 1:9]

    Av = nodeDiff(9)

    test = Av*d
    @test test==truth 
end 
    
function nodeDiff2D_test()

    n1,n2 =  3,2
    # Make the test data
    nodes = [1 2 3 4; 5 6 7 8; 9 10 11 12]'

    grad = zeros(n1*n2*2)

    grad = [ones(3*3), ones(2*4)*4]

    G = nodeDiff(n1,n2)

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

    n1,n2 = 1,1
    
    # Make an element, take the derivitive, then average the edge to test
    # dimensions
    cube = rand(n1+1,n2+1)

    G = nodeDiff(n1,n2)

    grad_cube = G*cube[:]

    AvE = edgeAvg(n1, n2)
    avg_grad_cube = AvE * grad_cube

    @test_approx_eq avg_grad_cube mean(grad_cube)
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
    

function helmholtz1D()

    # Just see if it runs to completion without error

    n1 = 10

    w_sqr = 1
    m = ones(n1)
    rho = ones(n1)
    q = zeros(n1+1)
    q[n1/2] = 1.0
        
    # Make all operators
    Av = nodeAvg(n1)
    G = nodeDiff(n1)

    # we are solving Au=q for u
    A = G'*diagm(rho)*G + w_sqr*diagm(Av'm)

    # solve it
    u = A\q;
end
    

function helmholtz2D()

    n1,n2 = 10,12

    w_sqr = 1
    m = ones(n1,n2)
    rho = ones(n1,n2)
    q = zeros(n1+1,n2+1)
    q[n1/2, n2/2] = 1.0
        
    # Make all operators
    Av = nodeAvg(n1,n2)
    AvE = edgeAvg(n1,n2)
    G = nodeDiff(n1,n2)

    # we are solving Au=q for u
    A = G'*diagm(AvE'*rho[:])*G + w_sqr*diagm(Av'm[:])

    # solve it
    u = A\q[:]

    u = reshape(u,n1+1, n2+1)
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
end 
    
nodeAvg1D_test()
nodeAvg2D_test()
nodeAvg3D_test()

edgeAvg2D_test()
edgeAvg3D_test()

nodeDiff1D_test()
nodeDiff2D_test()
nodeDiff3D_test()

helmholtz1D()
helmholtz2D()
helmholtz3D()
