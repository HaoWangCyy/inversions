include("operators.jl")

# Grid size constants
n1 = 10;
n2 = 11;
n3 = 12;

# Grid resolution constants
dn1 = .1;
dn2 = .2;
dn3 = .3;

# Make the known model params (constants to start)
w = ones(n1,n2,n3) * 10
k = ones(n1,n2,n3) * 5.0
sigma = ones(n1+1,n2+1,n3+1) * 2.0

# Make the operators
Av = AvOp(n1,n2,n3);
Avn = AvnOp(n1,n2,n3);
D = DiffOp(n1,n2,n3);
L_inv = ScaleOp(n1,n2,n3,1/dn1,1/dn2,1/dn3);

# Make the four blocks of our system matrix
#B1 = L_inv;
#B2 = D';
#B3 = -D* diagm(sigma[:])
#B4 = Av*L_inv*((w.*w.*k.*k)[:])

# A = [B1 B2;B3 B4];

A = D*spdiagm(Av*sigma[:])*D' + ((w[:].^2)*spdiagm(Avn*k[:].^2))
