using PyCall
@pyimport matplotlib.pyplot as plt

raw_data = readdlm("marmhard.dat")

# sampled at cell centers
n1, n2 = 122, 384
n_cells = n1*n2
n_nodes = (n1+1)*(n2+1)
data = float(raw_data[1:end-5])

# resolultion
d1 = 24.0 # m
d2 = 24.0 # m

# model parameters
m = 1. ./ data.^2
rho = ones(n_cells)

# source at every point
q = zeros((n1+1),(n2+1))
q = ones(n_nodes)

# Make all operators
Av = nodeAvg(n1,n2)
AvE = edgeAvg(n1,n2)
G = nodeDiff(n1,n2,d1,d2)
V = ones(n_cells) * d1*d2

# Plot the model
plt.figure()
plt.imshow(reshape(m, n1,n2), extent=[0,9192, 2904, 0])
plt.xlabel("offset [m]")
plt.ylabel("depth [m]")
plt.savefig("figures/figure7.eps")


freq = 2* pi *[1,10,100]
plt.figure()
plot = 310

for w in freq
    plot += 1
    # we are solving Au=-q for u
    LS = helmholtzNeumann(Av, AvE, G,V,rho, w^2, m)
    RS = -spdiagm(Av'*V)*q[:]
    u = LS\RS
    plt.subplot(plot)
    plt.imshow(reshape(u, n1+1, n2+1), extent=[0,9192, 2904, 0])
    plt.ylabel("depth [m]")

end
plt.xlabel("offset [m]")
plt.savefig(string("figures/figure8.eps"))
plt.show();

