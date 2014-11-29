using PyCall
include("solvers.jl")
include("operators.jl")
include("operators_test.jl")
include("inversion_lib.jl")

@pyimport matplotlib.pyplot as plt

# Define the model size
n1, n2 = 112,112

dv = 1./[112,112]

# PML params(top, bottom, left, right)
sigma = 100000.0
pad = (0,20,10,10)

# Make the model
m = ones(n1,n2)

# put in the target
m[20:30,50:60] = 2

# define the rest
w = 0.5

# Apply the PML to the model
S, m = PML(m, w, sigma, pad, dv)

#S = speye(size(S)[1])

rho = ones(size(m))

# make the q matrix
Q = zeros([size(m)...]+1..., n2+1)
for i=1:size(Q)[3]
    q = zeros([size(m)...]+1...)
    q[pad[1] + 1, pad[3]+i] = 1
    
    Q[:,:,i] = q
end


u = helmholtzNeumann(rho, w, m, Q, dv,S)

um = reshape(u[pad[1]+1:pad[1]+n1+1, pad[3]+1:pad[3]+n2+1, :],
             n1+1, n2+1, n2+1)

Dobs = real(reshape(um[1,:,:], n2+1, n2+1))



adjointVectorTest(u, H, P, w, dv)


plt.imshow(Dobs)
plt.show()



