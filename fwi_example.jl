using PyCall
include("solvers.jl")
include("operators.jl")
include("operators_test.jl")
include("inversion_lib.jl")

@pyimport matplotlib.pyplot as plt

# Define the model size
raw_data = readdlm("marmhard.dat")
data = reshape(float(raw_data[1:end-5]), 122,384)
m = 1. ./ data.^2
m0 = conv2(m, ones(10,10)/100)[10:end, 10:end]
# sampled at cell centers
n1, n2 = size(data)


dv = [24,24]

# PML params(top, bottom, left, right)
sigma = 10.0
pad = (0,40,20,20)

# Make the model

# put in the target
#m[40:60,20:30] = 3

# define the rest
w = 0.5

# Apply the PML to the model
m = pad_model(m, pad...)
m0 = pad_model(m0,pad...)
S, m = PML(m, w, sigma, pad, dv)
S0, m0 = PML(m0, w, sigma, pad, dv)
rho = ones(size(m))

# make the q matrix
Q = zeros([size(m)...]+1..., n2+1)
for i=1:size(Q)[3]
    q = zeros([size(m)...]+1...)
    q[pad[1] + 1, pad[3]+i] = 1
    
    Q[:,:,i] = q
end
P = reshape(Q, prod(size(Q)[1:2]), size(Q)[3])


# Solve the forward problem
u = helmholtzNeumann(rho, w, m, Q, dv,S)
um = reshape(u[pad[1]+1:pad[1]+n1+1, pad[3]+1:pad[3]+n2+1, :],
             n1+1, n2+1, n2+1)

Dobs = real(reshape(um[1,:,:], n2+1, n2+1))

# Test the sensitivity
A,dummy = helmholtz(rho,w,m,dv,S)
#adjointVectorTest(u, H, P, w, dv)


u = helmholtzNeumann(rho, w, m0, Q, dv,S)
um = reshape(u[pad[1]+1:pad[1]+n1+1, pad[3]+1:pad[3]+n2+1, :],
             n1+1, n2+1, n2+1)
D = real(reshape(um[1,:,:], n2+1, n2+1))

sig = 1e8
r = D - Dobs
mis = 0.5*sig*(r[:]'*r[:])
print(mis)
A,dummy = helmholtz(rho,w,m0,dv,S)
dmis = sig*jacobianTw(u, A, P, w, dv, r)
mc = m0
Ut=0
mt = 0
Dt=0
rt=0

for f=2:10
    w = 1/f
    m0 = pad_model(real(m0[pad[1]+1:end-pad[2],pad[3]+1:end-pad[4]]), pad...)
    S, m0 = PML(m0, w, sigma, pad, dv)
    u = helmholtzNeumann(rho, w, m0, Q, dv,S)
    um = reshape(u[pad[1]+1:pad[1]+n1+1, pad[3]+1:pad[3]+n2+1, :],
                 n1+1, n2+1, n2+1)
    D = real(reshape(um[1,:,:], n2+1, n2+1))
    
    sig = 1e8
    r = D - Dobs
    mis = 0.5*sig*(r[:]'*r[:])
    print(mis)
    A,dummy = helmholtz(rho,w,m0,dv,S)
    dmis = sig*jacobianTw(u, A, P, w, dv, r)
    
    for i = 1:25
        muLS = 1
        s = -dmis
        if maximum(abs(s))>0.1
            s = s/maximum(abs(s))*0.1
        end

        lscount = 1

        while true
            #mt,St = PML(real(mc +reshape(muLS*s, size(m))),w, sigma,pad,dv)
            mt = mc +reshape(muLS*s, size(m))
            Ut = helmholtzNeumann(rho, w, mt, Q, dv,S)
            um = reshape(Ut[pad[1]+1:pad[1]+n1+1, pad[3]+1:pad[3]+n2+1, :],
                         n1+1, n2+1, n2+1)
            Dt = real(reshape(um[1,:,:], n2+1, n2+1))
            rt = Dt - Dobs
            
            mist = 0.5*sig*(rt[:]'*rt[:])

            print(mist)

            if mist[1] < mis[1]
                break
            end 
            
            muLS = muLS/2
            lscount = lscount + 1
            if lscount > 6
                print("DAMN")
                break
            end
        end 
        mc = mt
        D = Dt
        U = Ut
        r = rt 
        ms = 0.5*sig*(r[:]'*r[:])
        A,dummy = helmholtz(rho,w,mc,dv,S)
        dmis = sig*jacobianTw(U, A, P, w, dv, r)


     
    end
    plt.subplot("221")
    plt.imshow(D)
    plt.subplot("222")
    plt.imshow(Dobs)
    #plt.imshow(real(mc[pad[1]+1:end-pad[2],pad[3]+1:end-pad[4]]))
    plt.subplot("223")
    plt.imshow(reshape(dmis, size(m)))
    plt.subplot("224")
    plt.imshow(r)
    

    m0 = mc
    
end
plt.show()
