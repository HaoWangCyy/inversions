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
    Av1 = kron(nodeAvg(n2),speye(n1))
    Av2 = kron(speye(n2),nodeAvg(n1))
    
    
    return [Av1 Av2] 
end 

function edgeAvg(n1,n2,n3)
    
    A1 = kron(nodeAvg(n3),kron(nodeAvg(n2),speye(n1))); 
    A2 = kron(nodeAvg(n3),kron(speye(n2),nodeAvg(n1))); 
    A3 = kron(speye(n3),kron(nodeAvg(n2),nodeAvg(n1)));
    
    return [A1 A2 A3]
end

function nodeDiff(n)
    # Returns a difference operator to differentiate on the nodes of a grid

    D =  spdiagm(tuple(ones(n) * -1, ones(n) *1),[0,-1])
    return D'
end


function nodeDiff(n1,n2)
    # Returns a difference operator to differentiate an n-element vector

    G1 = kron(speye(n2+1), nodeDiff(n1))
    G2 = kron(nodeDiff(n2), speye(n1+1))
    
    return [G1,G2]
end

function nodeDiff(n1,n2,n3)

    G1 = kron(speye(n3+1), kron(speye(n2+1), nodeDiff(n1)))
    G2 = kron(speye(n3+1), kron(nodeDiff(n2), speye(n1+1)))
    G3 = kron(nodeDiff(n3), kron(speye(n2+1), speye(n1+1)))


    return [G1,G2,G3]
end 

function helmholtzNeumann(Av, AvE, G, V, rho, w_sqr, m)
    # Returns the helmholtz operator for dirichlet boundary conditions.
   
    H = G'*diagm(AvE'*(rho.*V))*G + diagm(Av'*(w_sqr*V.*m))
    return H
end 
