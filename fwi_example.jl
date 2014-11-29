using PyCall
include("solvers.jl")
include("operators.jl")
include("inversion_lib.jl")

@pyimport matplotlib.pyplot as plt

# Define the model size
n1, n2 = 112,112
pad = (0,20,10,10)
dv = 1./[112,112]





# PML params(top, bottom, left, right)

sigma = 100000.0


# Make the model
m = ones(n1,n2)

# put in the target
m[20:30,50:60] = 2

# define the rest
w =1

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



H, dummy = helmholtz(rho, w, m, dv, S)

v = randn(size(m))*1e-4

V = randn(size(Q)[3], 113)*1e-4


P = reshape(Q,prod(size(Q)[1:2]),size(Q)[3])
PV = P*V
Z = (H')\PV

Gv1 = helmholtzDerivative(u[:,:,1],w,dv)*v[:]
Gv = zeros(Complex,length(Gv1), size(u)[3])
JTv = 0.0
for i = 2:n2+1

    
    Gi = helmholtzDerivative(u[:,:,i],w,dv)
    
    Gv[:,i] +=  Gi*v[:]
    JTv = JTv - Gi'*Z[:,i];
    
end 



Z = H\Gv
Jv = -real(P'*Z)
JTv = real(JTv)


JTv'*v[:] - V[:]'*Jv[:]






