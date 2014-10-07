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

    

nodeAvg1D_test()
nodeAvg2D_test()
