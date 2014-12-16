include("operators.jl")
include("inversion_lib.jl")

function helmholtzNeumann(rho, w, m, q, dv, S,s)
"""
Solves the helmholtz equation with a Neumann boundary conditions (the gradient
is zero at model boundaries). Solves for 1,2, and 3-D cases.

@param rho: The density of the earth model, sampled at the grid centers. Can be
            a single value of array the same size as m.
@param w: The radial frequency component to solve for (w=2pi f), sampled at
          the grid centers
@param m: The slowness profile of the model (1/c^2), sampled at the grid centers
@param q: The source term, sampled at grid nodes.
@param dv: A tuple of grid spacings for each dimension, ie: (dx,dy,dz).

@param S: The PML operating matrix
@param s: The PML scaling vector

@returns U: The wavefield solution
"""
    

    if ndims(q) == 3

        _q = reshape(q, prod(size(q)[1:2]), size(q)[3])
    else
        _q = q[:]
    end


    Av = nodeAvg(size(m)...)
    
    # Solve HU=Q
    H, Q = helmholtz(rho, w, m, dv, S, s)
    Q = Q*_q
  
    U = H\Q

    return reshape(U, size(q))
end 


function helmholtzSensitivity(P,rho, w, m, q, dv)

    H, Q = helmholtz(rho, w, m, dv)
    U = helmholtzNeumann(rho, w, m, q,dv)

    G = helmholtzDerivative(U,w,dv)

    J = -P*inv(full(H))*G

    return J
end
    

function gradientDescent(n, pml_sig, pml_pad, grad_sig, P, Q,
                         rho, m0, w, dv, plot_dir)
"""
Simple linear gradient descent algorithm with basic line seraching


@param n: Number of iterations of the descent
@param pml_sig: PML constant sigma
@param pml_pad: Padding used for PML (z1,z2,x1,x2)
@param grad_sig: Constant sigma for gradient scaling
@param P: The receiver sampling matrix
@param Q: Source sampling matrix
@param rho: Model density
@param m0: Starting model
@param w: Source frequency number
@param dv: Grid resolution (dz,dx)

@returns the best estimate model m
"""

    # rename some variables to accomodate copy and pasted code
    sigma = pml_sig
    sig = grad_sig
    pad = pml_pad
    nq = size(Q)[3]
    mc, Ia = pad_model(m0, pad...)


    # Initialize loop variables
    Ut=0;
    mt = 0;
    Dt=0;
    rt=0;

    
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


    all_misfit = zeros(n)
    # gradient iterations
    for i = 1:n
        muLS = 1;
        s = -dmis;


        # limit the gradient
        if maximum(abs(s))>0.1
            s = s/maximum(abs(s))*0.1 / w;
        end

        
        lscount = 1;
    

        while true

            mt = mc + reshape(real(Ia*muLS*s), size(mc))
            
            mt[real(mt).>4] = 4
            mt[real(mt).< 3] = 3
            
            Ut = helmholtzNeumann(rho, w, mt, Q, dv,S, s12);
        
            um = reshape(Ut, prod(size(u)[1:2]), nq);
            
            Dt = real(P'*um);

            rt = Dt - Dobs;
            
            mist = 0.5*sig*(rt[:]'*rt[:]);

            if mist[1] < mis[1]
                break
            end 
            
            muLS = muLS/2;
            lscount = lscount + 1;
            if lscount > 2
                print("DAMN")
                break
            end
       end

        mc = mt;

        D = Dt;
        U = Ut;
        r = rt ;
        
        ms = 0.5*sig*(r[:]'*r[:]);

        all_misfit[i] = ms[1];
        
        A,dummy = helmholtz(rho,w,mc,dv,S, s12);
        
        dmis = sig*jacobianTw(U, A, P, w, dv, r, s12, Ia);

        plt.figure()
        plt.imshow(reshape(dmis, size(m)))
        file_name = string(plot_dir,"/",w,"_",i, "_gradient", ".png")
        plt.savefig(file_name)

        
        dmis = dmis[:]
        
        plt.figure()
        plt.imshow(reshape(Ia'*mc[:], size(m)),vmin=2.5, vmax=4.5)
        file_name = string(plot_dir,"/",w,"_",i, "_model", ".png")
        plt.savefig(file_name)

        plt.close("all")

    end

    # Bring it back to regular dimensions
    return real(reshape(Ia'*mc[:], size(m0))), all_misfit
end
    


