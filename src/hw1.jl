#-----------------------------
#       Search Models
# Author: Dhananjay Ghei
# Date: October 29, 2018
#-----------------------------

# Loading the required libraries
using ForwardDiff
using NLSolversBase
using NLsolve
using SymPy
using LinearAlgebra
using QuantEcon
#------------------------------ Setting up the parameters
β, s̄, τ, x, ϕ, γ, α, δ, ρ, ζ, η, μss = .996, .0012, .4, .034, .5, .471, .33, .0028, .4, .00325, 1/2, 2.32

#------------------------------ Solving for the non stochastic steady state
kss, nss, css, wss, θss = Sym("kss", "nss", "css", "wss", "thetass")
eq1 = nss - μss*θss^(η)*(1-nss)-(1-x)*nss
eq2 = 1 - β*exp(-s̄/(1-α))*(α*((kss)/(nss-θss*(1-nss)))^(α-1)+(1-δ))
eq3 = kss*exp(s̄/(1-α)) - (1-δ)*kss + css - kss^(α)*(nss-θss*(1-nss))^(1-α)
eq4 = (1-τ)*wss - ϕ*(1+θss)*(1-α)*(1-τ)*((kss)/(nss-θss*(1-nss)))^α - (1-ϕ)*γ*css
eq5 = (1-α)*((kss)/(nss-θss*(1-nss)))^α - β*μss*θss^(η-1)*((1-α)*((kss)/(nss-θss*(1-nss)))^α *(1 + ((1-x)/(μss*θss^(η-1))))-wss)
# Steady states
sol = nsolve([eq1, eq2, eq3, eq4, eq5], [kss, nss, css, wss, θss], [218.2, 0.95, 4.6, 4.6, 0.078])
kss, nss, css, wss, θss = convert(AbstractFloat, sol[1]), convert(AbstractFloat, sol[2]), convert(AbstractFloat, sol[3]), convert(AbstractFloat, sol[4]), convert(AbstractFloat, sol[5])
#------------------------------ Log linearising around the steady state
# TODO
# Defining the functionals $C$ and $\Theta$
# Functional C
function CFun(s,n,k, Css, Cs, Cn, Ck)
    CFun = exp(log(Css)+ Cs*(s-s̄)+ Cn*(log(n)-log(nss)) + Ck * (log(k) - log(kss)))
    return CFun
end
# Functional θ
function ΘFun(s,n,k, θs, θn, θk)
    ΘFun = exp(log(θss) + θs*(s-s̄) + θn*(log(n) - log(nss)) + θk*(log(k)-log(kss)))
    return ΘFun
end
# Law of motion for states
function SFun(s)
    s1 = (1-ρ)*s̄ + ρ*s
    return s1
end

# Take the log linearised solutions directly from Shimer's book
# and proceed

# Defining the A and D matrix as in Shimer's book
A = [ρ 0 0; 0.026 0.312 -0.047; -0.605 0.019 0.991]
D = [ζ, 0, 0]
Iden = [1 0 0; 0 1 0; 0 0 1]

# Solving the discrete Lyapunov equation
tol = 1e-6
er = 1+tol
# Initial guess
Σ = D*D'
# Iterating
while er > tol
    Σt = A*Σ*A' + D*D'
    er = norm(Σ - Σt)
    Σ = Σt
end
Σ
# Run a sanity check to see if the values are okay
Σsanity = solve_discrete_lyapunov(A, D*D')

# Calculating the correlation between n and k
# Should be equal to -.999 as in Shimer's Table 3.2
# Indeed, it is -.999 as in Shimer's book
Σ[2,3]/(sqrt(Σ[2,2])*sqrt(Σ[3,3]))
# Next, we need to log linearise all the other variables
# We first start with the law of motion for employment
# Log linearising the employment level
# Defining m as in Shimer's book for \theta and c
Ã = [1.548 -0.48 -2.779; 0.381 0.014 0.603]
θCoeff = Ã[1,:]
cCoeff = Ã[2,:]

B = [ρ 0 0; 1.548 -0.480 -2.779; 0.381 0.014 0.603]
# Law of motion for employment
function employ(f)
    n,θ = f
    y = log(((1-.034)*nss*exp(n)+μss*θss^(η)*exp(η*θ)*(1-nss*exp(n)))/nss)
    return y
end
# At the steady state, it should return 0
employ([0,0])
# Find the Jacobian
employJac = ForwardDiff.gradient(employ, [0,0])
# Need to get the coefficients for the log linearised formula for n_t+1
nCoeff = employJac[2]*B[2,:] + [0, employJac[1], 0]
# They are the same as in Shimer's book Page 101

# Next, we log linearise the law of motion for capital
# Log linearising the law of motion for capital
function kPrime(x)
    s,n,k,θ,c = x
    y = log((kss^(α)*exp(α*k)*(nss*exp(n)-(1-nss*exp(n))*θss*exp(θ))^(1-α)+(1-δ)*kss*exp(k)-css*exp(c))*exp((-s̄*exp(s))/(1-α))/kss)
    return y
end
# At the steady state, it should return 0
kPrime([0,0,0,0,0])
# Find the Jacobian
kPrimeJac = ForwardDiff.gradient(kPrime, [0,0,0,0,0])
# Finding the coefficients for Capital tomorrow in terms of states (s,n,k)
# They are the same as in Shimer's book Page 101
kPrimeCoeff = [-(1-ρ)+kPrimeJac[1], kPrimeJac[2], kPrimeJac[3]] + kPrimeJac[4]*B[2,:] + kPrimeJac[5]*B[3,:]

# Next, we log linearise the output
# Log linearising the output
yss = exp(s̄)*(kss^(α))*(nss-(1-nss)*θss)^(1-α)
function gdp(x)
    s,n,k,θ = x
    y = log(exp(s̄*exp(s))*(kss^(α)*exp(α*k)*(nss*exp(n)-(1-nss*exp(n))*θss*exp(θ))^(1-α))/yss)
    return y
end
# At the steady state, it should return 0
gdp([0,0,0,0])
# Find the Jacobian
gdpJac = ForwardDiff.gradient(gdp, [0,0,0,0])
# Finding the coefficients for output in terms of states (s,n,k)
gdpCoeff = gdpJac[1:3] + gdpJac[4,]*Ã[1,:]

# Consumption - Output ratio
cyCoeff = cCoeff - gdpCoeff
# Log linearising wages
function wages(x)
    n,k,θ,c = x
    y = log((ϕ*(1+θss*exp(θ))*(1-α)*(1-τ)*((kss*exp(k))/(nss*exp(n)-θss*exp(θ)*(1-nss*exp(n))))^α + (1-ϕ)*γ*css*exp(c))/((1-τ)*(wss)))
    return y
end
# At the steady state, it should return 0
wages([0,0,0,0])
# Find the Jacobian
wagesJac = ForwardDiff.gradient(wages, [0,0,0,0])
# Finding the coefficients for wages in terms of states (s,n,k)
wageCoeff = [0, wagesJac[1], wagesJac[2]]+ wagesJac[3]*θCoeff + wagesJac[4]*cCoeff
# Finding the coefficients for wn/y (labor share) in terms of state (s,n,k)
wnyCoeff = wageCoeff + nCoeff - gdpCoeff
# ΣÃ = Ã*Σ*Ã'
# # Calculating the correlation between \theta and c
# # Should be equal to -.983 as in Shimer's Table 3.2
# # It is -.9811 which is pretty close to what Shimer had in his book
# ΣÃ[1,2]/sqrt(ΣÃ[1,1]*ΣÃ[2,2])
# Ã[2, ]

# Labor wedge
function wedge(x)
    n,c,y = x
    tau = log((1 - (((.513)/(1-α))*(css*exp(c)/(yss*exp(y)))*(nss*exp(n))))/τ)
    return tau
end

wedge([0,0,0])
wedgeJac = ForwardDiff.gradient(wedge, [0,0,0])
wedgeCoeff = [0, wedgeJac[1], 0] + wedgeJac[2]*cCoeff + wedgeJac[3]*gdpCoeff
# Stochastic term
sCoeff = [ρ, 0, 0]
# Finding the moments for all relevant variables
# y,c, θ, k, n, wn/y, c/y, ̂τ, s
# Setting up a big Ã matrix for correlations
Moments = [gdpCoeff'; cCoeff'; θCoeff'; kPrimeCoeff'; nCoeff'; wnyCoeff'; cyCoeff'; wedgeCoeff'; sCoeff']
# Calculating the moments of all the relevant variables
ΣMom= Moments*Σ*Moments'
# Relative standard deviation
sqrt.(diag(ΣMom)/ΣMom[1])
ΣMom[1,8]/sqrt(ΣMom[8,8]*ΣMom[1,1])
# Correlations
using DataFrames

#------------------------------ Generating Table 3.5 from Shimer's book

#------------------------------ Impulse response functions
# Setting up the matrix for the Blanchard-Kahn conditions
# D = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; Ã[2,:]' 0 0; Ã[1,:]' 0 0]
# G = [ρ 0 0 0 0; nCoeff' 0 0; kPrimeCoeff' 0 0; 0 0 0 0 0; 0 0 0 0 0]
# F = schur(D, G)
# eigen(pinv(D)*G)
