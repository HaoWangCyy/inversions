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



function nodeDiff(n)
    # Returns a difference operator to differentiate an n-element vector

    D =  spdiagm(tuple(ones(n) * 1, ones(n) *-1),[0,-1])
    return D'
end

