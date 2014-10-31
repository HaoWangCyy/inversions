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

# source at every point
q = ones(n1+1,n2+1)
#q = zeros(n1+1, n2+1)
#q[1,:] = -1.0


# Plot the model
plt.figure()
plt.imshow(reshape(m, n1,n2), extent=[0,9192, 2904, 0])
plt.xlabel("offset [m]")
plt.ylabel("depth [m]")
plt.savefig("figures/figure7.eps")


freq = 2* pi * [1:50]
plt.figure()
plot = 310

total = zeros(n1+1, n2+1)
for w in freq
    plot += 1

    u = helmholtzNeumann(rho, w, m, q, dv)

end

plt.imshow(total, extent=[0,9192, 2904, 0])
plt.ylabel("depth [m]")
plt.show()

plt.xlabel("offset [m]")
plt.savefig(string("figures/figure8.eps"))
plt.show();

