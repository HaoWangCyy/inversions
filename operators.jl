function Av(n)
    # Returns an averaging operator for averaging adjacent points in an n-element
    # vector
    
    Av = spdiagm(tuple(ones(n) * .5, ones(n) *.5),[0,-1])
    return Av
end

function Av(n1,n2)
    # Returns an averaging operator for averaging adjacent points in an mXn
    # matrix

    Av = kron(AvOp(n2), AvOp(n1))
    return Av
end

function Av(n1,n2,n3)
    # Returns an averaging operator for averaging adjacent points in an mXn
    # matrix

    Av = kron(AvOp(n3), AvOp(n1,n2))
    return Av
end

function Diff(n)
    # Returns a difference operator to differentiate an n-element vector

    D =  spdiagm(tuple(ones(n) * 1, ones(n) *-1),[0,-1])
    return D
end


    
function Diff(n1,n2)
    # Returns a difference operator for adjacent points

    G1 = kron(speye(n2+1),DiffOp(n1))
    G2 = kron(DiffOp(n2),speye(n1+1))
    return [G1';G2']
    
end

function Diff(n1,n2,n3)
    # Returns a difference operator for adjacent points

    G1 = kron(speye(n3+1),kron(speye(n2+1),DiffOp(n1))); 
    G2 = kron(speye(n3+1),kron(DiffOp(n2),speye(n1+1))); 
    G3 = kron(DiffOp(n3),kron(speye(n2+1),speye(n1+1)));
   
    
    return [G1';G2';G3']
end


