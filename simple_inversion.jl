using PyCall
include("solvers.jl")
include("operators.jl")
include("operators_test.jl")
include("inversion_lib.jl")

@pyimport matplotlib.pyplot as plt

n1 = 50;
n2 = 40;

#Pad and smooth models
m = repmat(linspace(1,4,n1+20),1,n2+20)


m0 = conv2(ones(20,20), m)[20:end, 20:end]/100.0;

# remove padding
m = m[1:n1,1:n2];
m0 = m0[1:n1, 1:n2];

#plt.subplot("211");
#plt.imshow(m);
#plt.subplot("212");
#plt.imshow(m0);quad 
#plt.show();

dv = [1/n1,1/n2];

# PML params(top, bottom, left, right)
sigma = 1e6;
pad = (10,10,10,10);

# define the rest
w = 5.0

# Apply the PML to the model
m_pad, Ia = pad_model(m, pad...);
m0_pad, dummy = pad_model(m0,pad...);
S, s12 = PML(m_pad, w, sigma, pad, dv);
S0, s0 = PML(m0_pad, w, sigma, pad, dv);


# Constant density
rho = ones(size(m_pad));

# make the q matrix
Q = zeros([size(m_pad)...]+1..., n2+1);
for i=1:size(Q)[3]
    q = zeros([size(m_pad)...]+1...)
    q[pad[1] + 1, pad[3]+i] +=1

    
    Q[:,:,i] = q;
end

# Receiver for every source
P = reshape(Q, prod(size(Q)[1:2]), size(Q)[3]);

#---------------------------------------------------------------#
# Solve the true forward problem
u = helmholtzNeumann(rho, w, m_pad, Q, dv,S, s12);
Dobs = real(P'*reshape(u, prod(size(u)[1:2]), size(u)[3]))

plt.imshow(Dobs)
plt.show()
#--------------------------------------------------------------#

# Test the sensitivity
A,dummy = helmholtz(rho,w,m_pad,dv, S, s12);
adjointVectorTest(u,m, A, P, w, dv, s12, Ia)
l, quad = jacobianConvergence(rho,w,m_pad,m,Q,P,A,dv,S,s12,Ia)



#--------------------------------------------------------------#


# Inversion through gradient descent---------------------------#

# Sove with m0
u = helmholtzNeumann(rho, w, m0, Q, dv,S,s);
um = reshape(u[pad[1]+1:pad[1]+n1+1, pad[3]+1:pad[3]+n2+1, :],
             n1+1, n2+1, n2+1);
D = real(reshape(um[1,:,:], n2+1, n2+1));

sig = 1.e1;
r = D - Dobs;
mis = 0.5*sig*(r[:]'*r[:]);

A,dummy = helmholtz(rho,w,m0,dv,S);
dmis = sig*jacobianTw(u, A, P, w, dv, r);

# initialize parameters used in the loop
mc = m0;
Ut=0;
mt = 0;
Dt=0;
rt=0;

# loop through frequencies
for f = w:w
    w = f;
    m0 = pad_model(real(m0[pad[1]+1:end-pad[2],pad[3]+1:end-pad[4]]), pad...);
    #S, m0 = PML(m0, w, sigma, pad, dv);
    u = helmholtzNeumann(rho, w, m0, Q, dv,S);
    um = reshape(u[pad[1]+1:pad[1]+n1+1, pad[3]+1:pad[3]+n2+1, :],
                 n1+1, n2+1, n2+1);
    D = real(reshape(um[1,:,:], n2+1, n2+1));
    
    r = D - Dobs;
    mis = 0.5*sig*(r[:]'*r[:]);
   
    A,dummy = helmholtz(rho,w,m0,dv,S);
    dmis = sig*jacobianTw(u, A, P, w, dv, r);

    
    dmis = reshape(dmis, size(m))
    dmis[1:pad[1],:] = 0
    dmis[end-pad[2]:end, :] = 0
    dmis[:,1:pad[3]] = 0
    dmis[:,end-pad[4]:end] = 0
    
    dmis = dmis[:]
 
    
    for i = 1:35
        muLS = 1;
        s = -dmis;
        
        if maximum(abs(s))>0.1
            s = s/maximum(abs(s))*0.1;
        end

        lscount = 1;

        while true
          
            mt = mc +real(reshape(muLS*s, size(m)));

            # Project onto solution space
            mt[real(mt).>4] = 4
            mt[real(mt).< 3] = 3
            Ut = helmholtzNeumann(rho, w, mt, Q, dv,S);
            um = reshape(Ut[pad[1]+1:pad[1]+n1+1, pad[3]+1:pad[3]+n2+1, :],
                         n1+1, n2+1, n2+1);
            Dt = real(reshape(um[1,:,:], n2+1, n2+1));
            rt = Dt - Dobs;
            
            mist = 0.5*sig*(rt[:]'*rt[:]);

            

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
        A,dummy = helmholtz(rho,w,mc,dv,S);
        dmis = sig*jacobianTw(U, A, P, w, dv, r);

        dmis = reshape(dmis, size(m))
        dmis[1:pad[1],:] = 0
        dmis[end-pad[2]:end, :] = 0
        dmis[:,1:pad[3]] = 0
        dmis[:,end-pad[4]:end] = 0

        dmis = dmis[:]

        print([ms, maximum(dmis)])
        plt.figure()
        plt.subplot("221")
        plt.imshow(D)
        plt.subplot("222")
        plt.imshow(Dobs)
        #plt.imshow(real(mc[pad[1]+1:end-pad[2],pad[3]+1:end-pad[4]]))
        plt.subplot("223")
        plt.imshow(reshape(dmis, size(m)))
        plt.subplot("224")
        plt.imshow(r)
       # plt.show()
     
    end
    plt.figure()
    plt.subplot("221")
    plt.imshow(D)
    plt.subplot("222")
    plt.imshow(Dobs)
    #plt.imshow(real(mc[pad[1]+1:end-pad[2],pad[3]+1:end-pad[4]]))
    plt.subplot("223")
    plt.imshow(reshape(dmis, size(m)))
    plt.subplot("224")
    plt.imshow(r)
    #plt.show()

    m0 = mc;
    
end

plt.figure()
plt.imshow(real(m0[pad[1]+1:end-pad[2], pad[3]+1:end-pad[4]]))
plt.show()
#plt.clf()
