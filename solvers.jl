include("operators.jl")

function helmholtzNeumann(rho, w, m, q, dv)
"""
Solves the helmholtz equation with a Neumann boundary conditions (the derivtives
are zero at model boundaries). Solves for 1,2, and 3-D cases.

@param rho: The density of the earth model, sampled at the grid centers. Can be
            a single value of array the same size as m.
@param w: The radial frequency component to solve for (w=2pi f), sampled at
          the grid centers
@param m: The slowness profile of the model (1/c^2), sampled at the grid centers
@param q: The source term, sampled at grid nodes.
@param dv: A tuple of grid spacings for each dimension, ie: (dx,dy,dz).

@returns U: The wavefield solution
"""
    

    if [size(q)...] != ([size(m)...] + 1)

        error("Invalid dimensions for source term q. Must be shape(m) +1")
    end



    Av = nodeAvg(size(m)...)
    # Solve HU=Q
    H, Q = helmholtz(rho, w, m, dv)
    Q = Q*q[:]

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
    
    
    


