using PyCall
include("solvers.jl")
include("operators.jl")
include("operators_test.jl")
include("inversion_lib.jl")

@pyimport matplotlib.pyplot as plt

raw_data = readdlm("marmhard.dat")
data = reshape(float(raw_data[1:end-5]), 122,384)
m  = 1 ./ data.^2

m = m ./ maximum(m)
m *= 4

smooth=30

m0 = conv2(ones(smooth,smooth), m)[smooth:end, smooth:end]/smooth^2;


m = m[1:end-smooth, 1:end-smooth]
m0 = m0[1:end-smooth, 1:end-smooth]

pad = (0,10,10,10);

m_pad = m
m0_pad = m0

m = m[:,150:225]
m0 = m0[:,150:225]

m_pad = m_pad[:, 150-pad[3]:225+pad[4]]
m0_pad = m0_pad[:, 150-pad[3]:225+pad[4]]
n1,n2=size(m)




#Pad and smooth models
#m = repmat(linspace(3,4,n1),1,n2);
#m = zeros(n1+smooth,n2+smooth)
#m[1:10,:] = 3.2
#m[11:20,:]  = 3.5
#m[21:30, :] = 3.8
#m[31:end,:] = 4.0




nq = n2
#plt.subplot("211");
#plt.imshow(m);
#plt.subplot("212");
#plt.imshow(m0);quad 
#plt.show();

dv = [1/n1,1/n2];

# PML params(top, bottom, left, right)
#sigma = 0;
#pad = (0,0,0,0);
sigma = 1e3;

# define the rest
w = 5.0

# Apply the PML to the model
dummy, Ia = pad_model(m, pad...);

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

plt.subplot("221")
plt.imshow(Dobs)
plt.subplot("222")
plt.imshow(real(u[:,:,1]))
plt.subplot("223")
plt.imshow(real(u[:,:,end]))
plt.subplot("224")
plt.imshow(m0) 
plt.show()
#--------------------------------------------------------------#

# Test the sensitivity
#A,dummy = helmholtz(rho,w,m_pad,dv, S, s12);
#adjointVectorTest(u,m, A, P, w, dv, s12, Ia)
#h,lin, quad = jacobianConvergence(rho,w,m_pad,m,Q,P,A,dv,S,s12,Ia)

#plt.plot(log(10,h),log(10,quad))
#plt.plot(log(10,h),log(10,lin))
#plt.show()

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
mc = m0_pad;
Ut=0;
mt = 0;
Dt=0;
rt=0;

# loop through frequencies
for f = 1:30
    w = f;
    sig = sig * 5/f
    S, s12 = PML(m_pad, w, sigma, pad, dv);
    u = helmholtzNeumann(rho, w, m_pad, Q, dv,S, s12);
    Dobs = real(P'*reshape(u, prod(size(u)[1:2]), size(u)[3]))
    

    S, s12 = PML(mc, w, sigma, pad, dv);
    Ut = helmholtzNeumann(rho, w, mc, Q, dv,S,s12);
    um = reshape(Ut, prod(size(u)[1:2]), nq);
    
    D = real(P'*um);
    
    r = D - Dobs;
    mis = 0.5*sig*(r[:]'*r[:]);
   
    A,dummy = helmholtz(rho,w,mc,dv,S,s12);

    dmis = sig*jacobianTw(u, A, P, w, dv, r, s12, Ia)
    dmis = dmis[:]

    # gradient iterations
    for i = 1:10
        muLS = 1;
        s = -dmis;

        # limit the gradient
        if maximum(abs(s))>0.1
            s = s/maximum(abs(s))*0.1;
        end

        
        lscount = 1;
    

        while true

            mt = mc + reshape(real(Ia*muLS*s), size(mc))
            
            #mt[real(mt).>4] = 4
            #mt[real(mt).< 3] = 3
            
            Ut = helmholtzNeumann(rho, w, mt, Q, dv,S, s12);
        
            um = reshape(Ut, prod(size(u)[1:2]), nq);
            
            Dt = real(P'*um);

            rt = Dt - Dobs;
            
            mist = 0.5*sig*(rt[:]'*rt[:]);

            print(mist)

            if mist[1] < mis[1]
                break
            end 
            
            muLS = muLS/2;
            lscount = lscount + 1;
            if lscount > 6
                print("DAMN")
                break
            end
       end

        mc = mt;

        D = Dt;
        U = Ut;
        r = rt ;
        ms = 0.5*sig*(r[:]'*r[:]);

        A,dummy = helmholtz(rho,w,mc,dv,S, s12);
        
        dmis = sig*jacobianTw(U, A, P, w, dv, r, s12, Ia);


        dmis = dmis[:]
        #plt.figure()
        #plt.subplot("221")
        #plt.imshow(D)
        #plt.subplot("222")
        #plt.imshow(Dobs)
        #plt.imshow(real(mc[pad[1]+1:end-pad[2],pad[3]+1:end-pad[4]]))
        #plt.subplot("223")
        #plt.imshow(reshape(dmis, size(m)))
        #plt.subplot("224")
        #plt.imshow(r)
        #plt.show()

        
    end
end 

m_end = real(reshape(Ia'*mc[:], size(m)))

plt.figure()
plt.subplot("221")
plt.imshow(m)
plt.subplot("222")
plt.imshow(m0)
plt.subplot("223")
plt.imshow(m_end)
plt.subplot("224")
plt.imshow(m - m_end)

plt.show()

print(norm(m-m_end))
