include("operators_test.jl")
using Base.Test

test_handler(r::Test.Success) = println("Success on $(r.expr)")
test_handler(r::Test.Failure) = println("test failed: $(r.expr)")
test_handler(r::Test.Error)   = println("test failed: $(r.expr)")

Test.with_handler(test_handler) do

nodeAvg1D_test()
nodeAvg2D_test()
nodeAvg3D_test()

edgeAvg2D_test()
edgeAvg3D_test()

nodeDiff1D_test()
nodeDiff2D_test()
nodeDiff3D_test()

helmholtz1D_converge()
helmholtz3D_converge()
helmholtz2D_converge()

end
