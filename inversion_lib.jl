include("operators.jl")
@pyimport matplotlib.pyplot as plt
function pad_model(m, p1,p2)

    padded_m = zeros([size(m)...] + [p1+p2]...)
    mask = zeros([size(m)...] + [p1+p2]...)

    padded_m[1:p1] = mean(m)
    
    padded_m[p1+1:end-p2] = m
    mask[p1+1:end-p2] = m
    
    padded_m[end-p2 + 1:end] = mean(m)

    return padded_m
end 

function pad_model(m, p1,p2,p3,p4)

    # Prevents the corners from being zero
    padded_m = ones([size(m)...] + [p1+p2,p3+p4]...)*mean(m)
    mask = ones([size(m)...] + [p1+p2,p3+p4]...)

    m_size = size(padded_m)
    padded_m[pad[1] + 1:m_size[1]-pad[2], pad[3]+1:m_size[2]-pad[4]] = m

    mask[pad[1] + 1:m_size[1]-pad[2], pad[3]+1:m_size[2]-pad[4]] = 0.0
    

    return padded_m
end 


function PML(m,w,sigma,pad,dv)

    padded_m = m

    if length(dv) == 1

        sigma_x = zeros(padded_m)
        sigma_x[1:pad[1]] = sigma*([1:pad[1]]*dv/pad[1]).^2
        sigma_x[end-pad[2]:end] = sigma*([1:pad[2]]*dv/pad[2]).^2
        S = 1 ./ (1+sigma_x./(w*padded_m)im)
        
    elseif length(dv) == 2

        
        sigma1 = zeros(padded_m)
        sigma1[1:pad[1],:] = repmat(sigma*([pad[1]:-1:1]*dv[1]/pad[1]).^2,1,
                                    size(padded_m)[2])
        sigma1[end-pad[2]+1:end,:] =  repmat(sigma*([1:pad[2]]*dv[1]/pad[2]).^2,
                                             1,
                                             size(padded_m)[2])

        s1  = 1. + (sigma1 ./ ((padded_m*w)im))
        
 

        sigma2 = zeros(padded_m)
        sigma2[:,1:pad[3]] = repmat(sigma*([pad[3]:-1:1]*dv[2]/pad[3]).^2,1,
                                    size(padded_m)[1])'
        sigma2[:,end-pad[4]+1:end] =  repmat(sigma*([1:pad[4]]*dv[2]/pad[4]).^2,
                                             1,
                                             size(padded_m)[1])'

        s2 = 1. + (sigma2 ./ ((padded_m*w)im))


        S = [(s2 ./ s1)[:] (s1 ./ s2)[:]]
        
        padded_m = s1 .* s2 .* padded_m

        
    elseif length(c)==3

        # Not implemented yet
        S=1
    end 

    S = centerToEdgeAvg(size(padded_m)...) * S[:]
    
    
    return spdiagm(S), padded_m

end
