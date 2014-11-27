using PyCall
include("solvers.jl")
include("inversion_lib.jl")

@pyimport matplotlib.pyplot as plt

raw_data = readdlm("marmhard.dat")


# sampled at cell centers
n1, n2 = 122, 384
n_cells = n1*n2
n_nodes = (n1+1)*(n2+1)
data = reshape(float(raw_data[1:end-5]), 122,384) 

# resolultion
dv = (24,24) #m

# PML settings
pad = (0,100,100,100)
C = (10000,10000)

# model parameters
m = 1. ./ data.^2
lambda = 100.0
k = 2*pi/lambda

S,m = PML(m, w, C, pad)

rho = ones(size(m))

# put sources on the surface
q = zeros([size(m)...]+1...)
q[pad[1]+1,[pad[2]+192, pad[2]+250]] = 100


u = helmholtzNeumann(rho, k, m, q, dv, S)
u_size = size(u)

u = u[pad[1] + 1:u_size[1]-pad[2], pad[3]:u_size[2]-pad[4]]



plt.imshow(real(u))
plt.xlabel("offset [m]")
plt.show()



