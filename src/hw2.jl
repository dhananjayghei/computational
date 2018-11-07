#------------------------------------
#       Business Cycle Accounting
# Author: Dhananjay Ghei
# Date: October 29, 2018
#------------------------------------

# Loading the required libraries
using ForwardDiff
using NLSolversBase
using NLsolve
using SymPy
using LinearAlgebra
using QuantEcon
using Plots
using DataFrames
using CSV

#----------------------------------------#
#      Setting up the parameters         #
#----------------------------------------#
β, δ, ψ, θ, γn, γz, σ = .975, .05, 2.5, .33, .010, .019, 1
τx, τl, zss, ḡ = .2,.3, 1, .5
kss, lss, css = Sym("kss", "lss","css")
eq1 = kss/lss - (((1+τx)*(1-β*(1+γn)*(1+γz)^(1-σ)*(1-δ)))/(β*(1+γn)*θ*zss^(1-θ)))^(1/(θ-1))
eq2 = css - ((kss/lss)^(θ-1)*zss^(1-θ)-(1+γn)*(1+γz)+1-δ)*kss+ḡ
eq3 = css -(((1-τl)*(1-θ)*(kss/lss)^(θ)*zss^(1-θ))/ψ)*(1-lss)
# Solving for the non-stochastic steady state
sol = nsolve([eq1, eq2, eq3], [kss, lss, css], [.7,.3,.56])
kss, lss, css = convert(AbstractFloat, sol[1]), convert(AbstractFloat, sol[2]), convert(AbstractFloat, sol[3])
#----------------------------------------#
#             Log linearisation          #
#----------------------------------------#
# Consumption
function CFun(x)
    g,kPrime,k,z,l = x
    y = log((((kss^(θ)*exp(θ*k))*((zss*lss)^(1-θ)*exp((1-θ)*z)*exp((1-θ)*l)))+(1-δ)*kss*exp(k)-(1+γn)*(1+γz)*kss*exp(kPrime)-ḡ*exp(g))/css)
    return y
end
CFun([0,0,0,0,0])
cCoeff = ForwardDiff.gradient(CFun, [0,0,0,0,0])
# Check the formulas from the Appendix (The values of the gradient match these values.)
# -(1+γn)*(1+γz)*kss/css
# -ḡ/css
# (kss^(θ)*(zss*lss)^(1-θ)*θ+(1-δ)*kss)/css
# (1-θ)*kss^(θ)*(zss*lss)^(1-θ)/css
css -(((1-τl)*(1-θ)*(kss/lss)^(θ)*zss^(1-θ))/ψ)*(1-lss)
# Labor
function LFun(x)
    c,k,z,τ,l = x
    y = ψ*css*exp(c)-(((1-τl*exp(τ))*(1-θ)*(kss^(θ)*exp(θ*k)*(lss^(-θ)*exp(-θ*l))*(zss^(1-θ)*exp((1-θ)*z)))*(1-lss*exp(l))))
    return y
end
LFun([0, 0, 0, 0, 0])
lCoeffInt = ForwardDiff.gradient(LFun, [0,0,0,0,0])
lCoeffInt[1:4] = lCoeffInt[1:4]/lCoeffInt[5]
clCoeff = push!(lCoeffInt[1]*cCoeff[1:4], 0) + [0, 0, lCoeffInt[2], lCoeffInt[3], lCoeffInt[4]]
# Log linearised labor in terms of g,kPrime,k,z,tau
lCoeff = clCoeff/(1-lCoeffInt[1]*cCoeff[5])

# ψ*css
# (1-θ)*(1-τl)*(kss^(θ)*lss^(-θ)*zss^(1-θ))*(1-lss)*(θ+(lss/(1-lss)))
# -(1-θ)*(1-τl)*(kss^(θ)*lss^(-θ)*zss^(1-θ))*(1-lss)*θ
# Output
yss = kss^(θ)*(zss*lss)^(1-θ)
function gdp(x)
    k,z,l = x
    y = log((kss^(θ)*exp(k)*(zss*lss)^(1-θ)*exp((1-θ)*z)*exp((1-θ)*l))/yss)
    return y
end
gdp([0,0,0])
# Log linearising output w.r.t k,z,l
gdpCoeff = ForwardDiff.gradient(gdp, [0,0,0])
# Log linearised output in terms of g,kPrime,k,z,tau
gdpCoeff[3]*lCoeff + pushfirst!(push!(gdpCoeff[1:2],0),0,0)
# Investment
xss = kss*(1+γn)*(1+γz) - (1-δ)*kss
function investment(x)
    kPrime, k = x
    y = log(((kss*exp(kPrime)*(1+γn)*(1+γz))-(1-δ)*kss*exp(k))/xss)
    return y
end
investment([0,0,0])
# Log linearising investment w.r.t kPrime and k
invCoeff = ForwardDiff.gradient(investment, [0,0])

# Euler equation
