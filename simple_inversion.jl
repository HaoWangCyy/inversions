using PyCall
include("solvers.jl")
include("operators.jl")
include("operators_test.jl")
include("inversion_lib.jl")

plot_dir = "app_figures2"

@pyimport matplotlib.pyplot as plt

n1 = 50;
n2 = 40;
nq = n2;

smooth=30

#Pad and smooth models
m = zeros(n1+smooth,n2+smooth)
m[1:10,:] = 3.2
m[11:20,:]  = 3.5
m[21:30, :] = 3.8
m[31:end,:] = 4.0

m = m[1:n1,1:n2];
m0 = ones(size(m))*mean(m);




plt.figure()
plt.imshow(m, vmin=2.5,vmax=4.5)
plt.savefig("true_model.png")

plt.figure()
plt.imshow(m0, vmin=2.5,vmax=4.5)
plt.savefig("initial_model2.png")

plt.close("all")


dv = [1/n1,1/n2];

# PML params(top, bottom, left, right)
sigma = 1e5;
pad = (30,30,30,30);

# define thetest w
w = 30.0

# Apply the PML to the model
m_pad, Ia = pad_model(m, pad...);
m0_pad, dummy = pad_model(m0,pad...);
S, s12 = PML(m_pad, w, sigma, pad, dv);
S0, s0 = PML(m0_pad, w, sigma, pad, dv);


# Constant density
rho = ones(size(m_pad));

# make the q matrix
Q = zeros([size(m_pad)...]+1..., nq);
for i=1:size(Q)[3]
    q = zeros([size(m_pad)...]+1...)
    q[pad[1] + 1, pad[3]+i+2] +=1

    
    Q[:,:,i] = q;
end

# Receiver for every source
P = reshape(Q, prod(size(Q)[1:2]), size(Q)[3]);

#---------------------------------------------------------------#
# Solve the true forward problem
u = helmholtzNeumann(rho, w, m_pad, Q, dv,S, s12);
um = real(reshape(u, prod(size(u)[1:2]), size(u)[3]));
Dobs = real(P'*um)

#plt.subplot("221")
#plt.imshow(Dobs)
#plt.subplot("222")
#plt.imshow(real(u[:,:,1]))
#plt.subplot("223")
#plt.imshow(real(u[:,:,end]))
#plt.subplot("224")
#plt.imshow(m0) 
#plt.show()
#--------------------------------------------------------------#

# Test the sensitivity
#A,dummy = helmholtz(rho,w,m_pad,dv, S, s12);
#adjointVectorTest(u,m, A, P, w, dv, s12, Ia)
#h,lin, quad = jacobianConvergence(rho,w,m_pad,m,Q,P,A,dv,S,s12,Ia)


#plt.figure()
#plt.plot(h,quad, label="2nd order")
#plt.plot(h, h.^2, label="quadratic")
#plt.plot(h,lin, label="1st order")
#plt.plot(h,h, label="linear")
#plt.xscale("log",basex=10)
#plt.yscale("log", basey=10)
#plt.legend(loc=4)
#plt.xlabel("h")
#plt.ylabel("difference")

#plt.savefig("derivative_test.png")
#plt.close("all")

#--------------------------------------------------------------#


# Inversion through gradient descent---------------------------#

# Sove with m0
u = helmholtzNeumann(rho, w, m0_pad, Q, dv,S,s12);

D = real(P'*reshape(u, prod(size(u)[1:2]),  nq));

sig = 1.e9;
r = D - Dobs;
mis = 0.5*sig*(r[:]'*r[:]);

A,dummy = helmholtz(rho,w,m0_pad,dv,S,s12);
dmis = sig*jacobianTw(u, A, P, w, dv, r,s12,Ia);

# initialize parameters used in the loop
mc = m0;

all_misfit = Float32[]

# loop through frequencies
for f = 1:20
    
    w = f;
    sig_new = sig * 5/f;
    n = 10;
    
    mc, mf = gradientDescent(n, sigma, pad, sig_new, P, Q,
                             rho, mc, w, dv, plot_dir);
        
 push!(all_misfit, mf...);
    
end 


m_end = mc;

writedlm("misfit.txt", all_misfit)

#plt.figure()
#plt.subplot("221")
#plt.imshow(m)
#plt.subplot("222")
##plt.imshow(m0)
#plt.subplot("223")
#plt.imshow(m_end)
#plt.subplot("224")
#plt.imshow(m - m_end)

#plt.show()

#print(norm(m-m_end))
