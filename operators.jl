function nodeAvg(n)
    # Returns an averaging operator for averaging nodes to grid centers
    
    Av = spdiagm(tuple(ones(n) * .5, ones(n) *.5),[0,-1])
    return Av'
end

    
function nodeAvg(n1,n2)
    # Returns an averaging operator for averaging nodes to grid centers

    Av = kron(nodeAvg(n2), nodeAvg(n1))
    return Av
end

function nodeAvg(n1,n2,n3)
    # Returns an averaging operator for averaging nodes to grid centers

    Av = kron(nodeAvg(n3), nodeAvg(n1,n2))
    return Av
end

function edgeAvg(n)
    return speye(n)
end

function edgeAvg(n1,n2)
    # Averages from edges to cell centers

    Av1 = kron(nodeAvg(n2), speye(n1))
    Av2 = kron(speye(n2), nodeAvg(n1))
 
    return [Av1 Av2] 
end 

function edgeAvg(n1,n2,n3)
    
    A1 = kron(nodeAvg(n3),kron(nodeAvg(n2),speye(n1))); 
    A2 = kron(nodeAvg(n3),kron(speye(n2),nodeAvg(n1))); 
    A3 = kron(kron(speye(n3), nodeAvg(n1)),nodeAvg(n2));
    
    return [A1 A2 A3]
end


function centerToEdgeAvg(n1)
    # Averages from cell center cell edges

    Av = spdiagm(tuple(ones(n1)*.5, ones(n1)*.5),[0,-1])

    Av[1,1] = 1.0
    Av[end,end] = 1.0

    return Av
end


function centerToEdgeAvg(n1,n2)

    A1 = kron(centerToEdgeAvg(n1), speye(n2))
    A1 = [A1 spzeros(size(A1)...)]
    
    A2 = kron(speye(n1), centerToEdgeAvg(n2))
    A2 = [spzeros(size(A2)...) A2]

    return [A1, A2]
end

function centerToEdgeAvg(n1,n2,n3)
    
    A1 = kron(centerToEdgeAvg(n3),kron(centerToEdgeAvg(n1), speye(n2)))

    A2 = kron(centerToEdgeAvg(n3),kron(speye(n2),centerToEdgeAvg(n1))); 

    A3 = kron(kron(speye(n3), centerToEdgeAvg(n1)),centerToEdgeAvg(n2));
    
    return [A1, A2, A3]
end



function nodeDiff(n, dn)
    # Returns a difference operator to differentiate on the nodes of a grid

    D =  spdiagm(tuple(ones(n) * -1, ones(n) *1),[0,-1])/dn
    return D'
end


function nodeDiff(n1,n2, dn1, dn2)
    # Returns a difference operator to differentiate an n-element vector

    
    G1 = kron(speye(n2+1), nodeDiff(n1, dn1))
    G2 = kron(nodeDiff(n2, dn2), speye(n1+1))

    
    return [G1,G2]
end

function nodeDiff(n1,n2,n3,d1,d2,d3)

    G1 = kron(speye(n3+1), kron(speye(n2+1), nodeDiff(n1, d1)))
    G2 = kron(speye(n3+1), kron(nodeDiff(n2,d2), speye(n1+1)))
    G3 = kron(nodeDiff(n3,d3), kron(speye(n1+1), speye(n2+1)))

    return [G1,G2,G3]
end 

        
                 
function helmholtz(rho, w, m, dv, S=S)
"""
Makes the helmoltz operators H and Q, HU=Qq
"""

    isdefined(:S) || (S=1) 
    
    n_cells = size(m[:])
    
    # Make the operators
    V  = ones(n_cells...)*prod(dv)
    Av = nodeAvg(size(m)...)
    AvE = edgeAvg(size(m)...)
    G = nodeDiff(size(m)...,dv...)
    
    H = -G'*spdiagm(AvE'*(rho[:].*V))*(S * G) + spdiagm(Av'*((w^2)*V.*m[:]))
    #H = -G'*S *G + spdiagm(Av'*((w^2)*V.*m[:]))
    Q = -spdiagm(Av'*V)
    
    return H, Q
end

function jacobianTw(u, A, P, w, dv, V)

    PV = P*V
    Z = (A')\PV

    jTv = 0
    for i = 1:size(P)[2]

        Gi = helmholtzDerivative(u[:,:,i],w,dv)
        jTv -= Gi'*Z[:,i]
    end 


    return real(jTv)
end 

function jacobianv(u, A, P, w, dv, v)


    Gv1 = helmholtzDerivative(u[:,:,1],w,dv)*v[:]
    Gv = zeros(Complex,length(Gv1), size(u)[3])

    for i = 1:size(P)[2]

        Gi = helmholtzDerivative(u[:,:,i],w,dv)
        
        Gv[:,i] +=  Gi*v[:]
        
    end
    
    Z = A\Gv 
    return -real(P'*Z)

end

function helmholtzDerivative(U,w,dv,pad=pad)


    Av = nodeAvg([size(U)...]-1 ...)
    
    n_cells = prod([size(U)...]-1)
    v = ones(n_cells)*prod(dv)

    G = w^2 * spdiagm(U[:])*Av'*spdiagm(v)

    print(size(G))
    return G
end 

    
