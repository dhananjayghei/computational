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
# Will do log-linearizing later

# Take the log linearised solutions directly from Shimer's book
# and proceed

s̄ + (ζ/sqrt(1-ρ^2))

# Defining the A and D matrix as in Shimer's book
A = [ρ 0 0; 0.026 0.312 -0.047; -0.605 0.019 0.991]
D = [ζ, 0, 0]
Iden = [1 0 0; 0 1 0; 0 0 1]
(Iden-A*A')^(-1)*(D*D')

tol = 1e-6
er = 1+tol

Σ = D*D'

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
# Need to get the coefficients for the log linearised formula for k_t+1
# They are the same as in Shimer's book Page 101
[-(1-ρ)+kPrimeJac[1], kPrimeJac[2], kPrimeJac[3]] + kPrimeJac[4]*B[2,:] + kPrimeJac[5]*B[3,:]

# Next, we log linearise the output
# Log linearising the output
yss = exp(s̄)*(kss^(α))*(nss-(1-nss)*θss)^(1-α)
function gdp(x)
    s,n,k,θ = x
    y = log(exp(s̄*exp(s))*(kss^(α)*exp(α*k)*(nss*exp(n)-(1-nss*exp(n))*θss*exp(θss))^(1-α))/yss)
    return y
end
# At the steady state, it should return 0
gdp([0,0,0,0])
# Find the Jacobian
gdpJac = ForwardDiff.gradient(gdp, [0,0,0,0])

# Defining m as in Shimer's book for \theta and c
Ã = [1.548 -0.48 -2.779; 0.381 0.014 0.603]
ΣÃ = Ã*Σ*Ã'
# Calculating the correlation between \theta and c
# Should be equal to -.983 as in Shimer's Table 3.2
# It is -.9811 which is pretty close to what Shimer had in his book
ΣÃ[1,2]/sqrt(ΣÃ[1,1]*ΣÃ[2,2])

#------------------------------ Impulse response functions
