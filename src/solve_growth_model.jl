# ------------------------------------------------#
# This is a function file
# designed to solve for the Prototype growth model
# using Modified Vaughan's Method
# ------------------------------------------------#

using NLsolve
using ForwardDiff
using SymPy
using LinearAlgebra
#=
@author : Dhananjay Ghei
@date : 2018-11-18
=#
@doc doc"""
This function file solves the prototype growth model

The prototype growth model is the one as in Ellen's Homework 2
"""

"""
##### Fields
- `calibrated::Vector`:Vector containing β, δ, ψ, γn, γz, σ
- `estimated::Vector`:Vector containing τx, τl, zss, ḡ
"""
mutable struct Parameters
    calibrated::Array
    estimated::Array
end

"""
Main constructor for parameter type

##### Arguments
- `β::Float64`  : Discount factor
- `δ::Float64`  : Depreciation
- `ψ::Float64`  : Disutility from labor
- `γn::Float64` : Population growth rate
- `γz::Float64` : Trend growth rate
- `σ::Float64`  : Coefficient of relative risk aversion
- `τx::Float64` : Tax rate on investment
- `τl::Float64` : Tax rate on labor income
- `zss::Float64`: Technology at steady state
- `ḡ::Float64`  : Steady state government spending

##### Returns
- `params::Parameters` : Parameters as calibrated and estimated of type Parameters
"""
function _set_params(β, δ, ψ, θ, γn, γz, σ, τx, τl, zss, ḡ)
    calibrated = [β, δ, ψ, θ, γn, γz, σ]
    estimated = [τx, τl, zss, ḡ]
    return Parameters(calibrated, estimated)
end

# params = _set_params(.975, .05, 2.5, .33, .010, .019, 1, .2, .3, 1, .5)

"""
Solves for the non-stochastic steady state

##### Arguments
- `params::Parameters` : parameters of Parameters type

##### Returns
- kss::Float64 : Capital at the steady state
- lss::Float64 : Hours worked at the steady state
- css::Float64 : Consumption at the steady state
- yss::Float64 : Output at the steady state
- xss::Float64 : Investment at the steady state
"""
function _solve_steady_state(params::Parameters)
    β, δ, ψ, θ, γn, γz, σ = params.calibrated
    τx, τl, zss, ḡ = params.estimated
    kss, lss, css = Sym("kss", "lss","css")
    eq1 = kss/lss - (((1+τx)*((1+γn)*(1+γz)-β*(1+γn)*(1+γz)^(1-σ)*(1-δ)))/
        (β*(1+γn)*(1+γz)^(1-σ)*θ*zss^(1-θ)))^(1/(θ-1))
    eq2 = css - ((kss/lss)^(θ-1)*zss^(1-θ)-(1+γn)*(1+γz)+1-δ)*kss+ḡ
    eq3 = ψ*css/(1-lss) -((1-τl)*(1-θ)*(kss/lss)^(θ)*zss^(1-θ))
    # Solving for the non-stochastic steady state
    sol = nsolve([eq1, eq2, eq3], [kss, lss, css], [.7,.3,.56])
    kss, lss, css = convert(AbstractFloat, sol[1]), convert(AbstractFloat, sol[2]), convert(AbstractFloat, sol[3])
    yss = zss*(kss)^(θ)*(lss)^(1-θ)
    xss = kss*(1+γn)*(1+γz) - (1-δ)*kss
    return kss, lss, css, yss, xss
end

# # Solve for non-stochastic steady state
# kss, lss, css, yss, xss = _solve_steady_state(params)
# ss = [kss, lss, css, yss, xss]

# ss = _solve_steady_state(params)
# typeof(ss)
# Homework 1 - Ellen works
# paramHW1 = _set_params(.95,.1,1.7,1/3,.03,.05,1,0,0,1,0)
# _solve_steady_state(paramHW1)

"""
Consumption as a function of resource constraint (in logs)

##### Arguments
- `x::Vector` : Vector of
- `ss_values::Vector` : Vector containing the steady state values
- `params::Parameters` : Vector of parameters of type Parameters

##### Return
- `c::Scalar` : Scalar log value of consumption
"""
function consumption(x::Vector, ss_values::Vector, params::Parameters)
    β, δ, ψ, θ, γn, γz, σ = params.calibrated
    τx, τl, zss, ḡ = params.estimated
    kss, lss, css, yss, xss = ss_values
    g, kPrime, k, z, l = x
    y = log((((kss^(θ)*exp(θ*k))*((zss*lss)^(1-θ)*exp((1-θ)*z)*exp((1-θ)*l)))+(1-δ)*kss*exp(k)-(1+γn)*(1+γz)*kss*exp(kPrime)-ḡ*exp(g))/css)
    return y
end

"""
Labor from the consumption-leisure equation (in logs)

##### Arguments
- `x::Vector` : Vector of
- `ss_values::Vector` : Vector containing the steady state values
- `params::Parameters` : Vector of parameters of type Parameters

"""
function labor(x::Vector, ss_values::Vector, params::Parameters)
    β, δ, ψ, θ, γn, γz, σ = params.calibrated
    τx, τl, zss, ḡ = params.estimated
    kss, lss, css, yss, xss = ss_values
    c, k, z, τ, l = x
    y = ψ*css*exp(c)-(((1-τl*exp(τ))*(1-θ)*(kss^(θ)*exp(θ*k)*(lss^(-θ)*exp(-θ*l))*
            (zss^(1-θ)*exp((1-θ)*z)))*(1-lss*exp(l))))
    return y
end

"""
Investment from the law of motion for capital

##### Arguments
- `x::Vector` : Vector of kPrime (Capital tomorrow) and k (Capital today)
- `ss_values::Vector` : Vector containing the steady state values
- `params::Parameters` : Vector of parameters of type Parameters

"""
function investment(x, ss_values::Vector, params::Parameters)
    β, δ, ψ, θ, γn, γz, σ = params.calibrated
    τx, τl, zss, ḡ = params.estimated
    kss, lss, css, yss, xss = ss_values
    kPrime, k = x
    y = log(((kss*exp(kPrime)*(1+γn)*(1+γz))-(1-δ)*kss*exp(k))/xss)
    return y
end

"""
Calculates the output (in logs)

##### Arguments
- `x::Vector`         : Vector of functions
- `ss_values::Vector` : Vector of steady states
- `params::Parameters`: parameters of the Parameters type

##### Returns
- y::Scalar : log output deviation from the steady state
"""
function gdp(x::Vector, ss_values::Vector, params::Parameters)
    β, δ, ψ, θ, γn, γz, σ = params.calibrated
    τx, τl, zss, ḡ = params.estimated
    kss, lss, css, yss, xss = ss_values
    k, z, l = x
    y = log((kss^(θ)*exp(θ*k)*(zss*lss)^(1-θ)*exp((1-θ)*z)*exp((1-θ)*l))/yss)
    return y
end

"""
Calculates the residual from Euler equation

##### Arguments
- `x::Vector`         : Vector of functions
- `ss_values::Vector` : Vector of steady states
- `params::Parameters`: parameters of the Parameters type

##### Returns
- y::Scalar
"""
function eulerEQ(x, ss_values::Vector, params::Parameters)
    β, δ, ψ, θ, γn, γz, σ = params.calibrated
    τx, τl, zss, ḡ = params.estimated
    kss, lss, css, yss, xss = ss_values
    c,cPrime,kPrime,zPrime,τX,τXPrime,l,lPrime = x
    y = (1+τx*exp(τX))*(1+γn)*(1+γz)*(css^(-σ)*exp(-σ*c))*(1-lss*exp(l))^(ψ*(1-σ)) -
        β*(1+γn)*(1+γz)^(1-σ)*(css^(-σ)*exp(-σ*cPrime))*(1-lss*exp(lPrime))^(ψ*(1-σ))*
        (θ*kss^(θ-1)*exp((θ-1)*kPrime)*(zss^(1-θ)*exp((1-θ)*zPrime))*(lss^(1-θ)*exp((1-θ)*
        lPrime)) + (1-δ)*(1+τx*exp(τXPrime)))
  return y
end

"""
Log linearizes the equations of growth model around the steady state

##### Arguments
- `ss_values::Vector` : Vector of steady states
- `params::Parameters`: parameters of the Parameters type

##### Returns
- `cCoeff::Vector` : Vector of coefficients for consumption
- `lCoeff::Vector` : Vector of coefficients for labor
- `yCoeff::Vector` : Vector of coefficients for output
- `xCoeff::Vector` : Vector of coefficients for investment
"""
function _log_linearize(ss_values::Vector, params::Parameters)
    # Log linearizing the resource constraint
    cCoeff = ForwardDiff.gradient(x -> consumption(x, ss_values, params), zeros(5))
    # Log linearizing the consumption-leisure equation
    lCoeff = ForwardDiff.gradient(x -> labor(x, ss_values, params), zeros(5))
    eulerCoeff = ForwardDiff.gradient(x -> eulerEQ(x, ss_values, params), zeros(8))
#    kPrimeCoeff = ForwardDiff.gradient(x -> eulerEQ(x, ), zeros(8))
    # Log linearizing the output equation
    yCoeff = ForwardDiff.gradient(x -> gdp(x, ss_values, params), zeros(3))
    # Log linearizing the investment equation
    xCoeff = ForwardDiff.gradient(x -> investment(x, ss_values, params), zeros(2))
    return cCoeff, lCoeff, eulerCoeff, yCoeff, xCoeff
end

"""
Computes the a and b coefficients of two linear expectational difference equations
TODO - Fix the arguments and make it more elegant
"""
function _prepare_modified_vaughan(cCoeff, lCoeff, eulerCoeff)
    # DO NOT LIKE THIS - BUT, IT IS A QUICK FIX for the time being
    # Get the coefficients for the first expectational equation ('a' coefficients)
    # Consumption-Leisure trade-off
    lNew = lCoeff[1]*cCoeff + [0, 0, lCoeff[2], lCoeff[3], lCoeff[5]]
    a = [lNew[3], lNew[2], lNew[5], lNew[4], lCoeff[4], lNew[1]]
    # Get the coefficients for the second expectational equation
    # Euler equation
    # Eliminating c and cPrime from the equation
    temp1 = eulerCoeff[1]*cCoeff
    temp2 = eulerCoeff[2]*cCoeff
    kPC = temp1[2] + temp2[3] + eulerCoeff[3]
    zPC = temp2[4] + eulerCoeff[4]
    lPC = temp2[5] + eulerCoeff[8]
    lC = temp1[5] + eulerCoeff[7]
    b = [temp1[3], kPC, temp2[2], lC, lPC, temp1[4], eulerCoeff[5], temp1[1], zPC,
        eulerCoeff[6], temp2[1]]
    return a, b
end

# cCoeff, lCoeff, eulerCoeff = dumb[1], dumb[2], dumb[3]
# a, b = _prepare_modified_vaughan(cCoeff, lCoeff, eulerCoeff)

"""
Wrapper function that solves for the steady state, log linearizes the system around a steady state, and solves the growth model using Modified Vaughan's method

##### Arguments
- `params::Parameters` : Declare parameters of type Parameters
"""
function _solve_growth_model(params::Parameters)
    # Solve for the non-stochastic steady state
    ss = _solve_steady_state(params)
    # Get log-linearized coefficients
    coeffs = _log_linearize(ss, params)
    # Modified Vaughan's method
    # Step 1: Find the generalised eigenvalues and eigenvectors
    evalues, evectors = eigen(A1, -A2)
    # Step 2: Sort the eigenvalues and eigenvectors so that they are in (1,1) position
    indx = sortperm(-evalues) # Descending order
    Λ = evalues[indx] # Re-arranging eigenvalues
    V = evectors[:,indx] # Re-arranging eigenvectors
    Λ = Diagonal(Λ)
    # Step 3: Calculate A and C
    A = V[1,1]*Λ[1,1]^(-1)*V[1,1]^(-1)
    C = V[2:end,1]*V[1,1]^(-1)
    # Step 4: Solve for B and D
    function findBD!(eq, x::Vector)
        B = x[1:4]'
        D2 = x[5:8]'
        eq[1:4] = a[2]*B + a[3]*D2 + [a[4] a[5] 0 a[6]]
        eq[5:8] = b[2]*B + b[3]*A*B + b[3]*B*P + b[4]*D2 + b[5]*c[2]*B + b[5]*B*P + [b[6] 0 b[7] b[8]] + [b[9] 0 b[10] b[11]]*P
        return eq
    end
    solBD = nlsolve(findBD!, ones(8), ftol = :1.0e-20, method = :trust_region , autoscale = true)
    B, D = solBD.zero[1:4], solBD.zero[5:8]
    return A,B,C
end

# function _kalman_filter(A, B, C, R)
#     # Initialize x_hat_0
# end

"""
Utility function

##### Arguments
- `x::Vector` : Vector of consumption and labor
- `params::Parameters` : parameters of Parameters type

##### Returns
- Utility for a given level of consumption and labor
"""
function u(x::Vector, params::Parameters)
    β, δ, ψ, θ, γn, γz, σ = params.calibrated
    c, l = x
    if σ==1.
        return log(c) + ψ * log(1-l)
    else
        return (c*(1-l)^ψ)^(1-σ)/(1-σ)
    end
end
