using PyCall
include("solvers.jl")
@pyimport matplotlib.pyplot as plt

raw_data = readdlm("marmhard.dat")

# sampled at cell centers
n1, n2 = 122, 384
n_cells = n1*n2
n_nodes = (n1+1)*(n2+1)
data = reshape(float(raw_data[1:end-5]), 122,384)

# resolultion
dv = (24,24) #m

# model parameters
m = 1. ./ data.^2
rho = ones(size(data))

# put sources on the surface
q = zeros(n1+1, n2+1)
q[10,180] = .001


# Plot the model
plt.figure()
plt.imshow(reshape(m, n1,n2), extent=[0,9192, 2904, 0])#
plt.xlabel("offset [m]")
plt.ylabel("depth [m]")
plt.show()plt
plt.savefig("figures/figure7.eps")


freq = [i for i in 1:10]

time = linspace(0,1,1000)

u = zeros(1000,385)
for f in freq
    
    
    w = 2* pi * f
    
    H = helmholtzNeumann(rho, w, m, q, dv)

    #uf = cos(w*time) .* H[1,:]
    u = H
end

plt.imshow(u)
plt.xlabel("offset [m]")
plt.show()


