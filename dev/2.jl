using SciMLSensitivity, OrdinaryDiffEq
using QuadGK, Enzyme, Zygote, ForwardDiff, LinearAlgebra
using Test
# Enzyme.runtimeActivity!(true)
include("main_pkgs.jl")

function Base.zero(nn::NN)
  newnn = deepcopy(nn)
  for l in newnn.layers
    l.W .= 0.0
    l.b .= 0.0
  end
  for inter in newnn.intermediates
    inter .= 0.0
  end
  return newnn
end

function applydense!(d::Dense, inp, out)
  mul!(out, d.W, inp, 1.0, 0.0)
  for i in eachindex(out)
    out[i] += d.b[i]
  end
  nothing
end

function applyNN!(nn::NN{T}, inp, out::AbstractArray{T}) where {T}
  applydense!(nn.layers[1], inp, nn.intermediates[1])
  for i in eachindex(nn.layers)[2:end]
    applydense!(nn.layers[i], nn.intermediates[i-1], nn.intermediates[i])
  end
  out .= nn.intermediates[end]
  nothing
end

const step = 0.22
const H_0 = diagm(1:2)
const H_D = 0.01 * [0.0 1.0; 1.0 0.0]
const repart = 1:2
const impart = 3:4

##cell

function make_dfunc(T)
  nn = NN(4, [Dense(4, 10, tanh, T), Dense(10, 4, sin, T)], T)
  plen = paramlength(nn)
  set_params(nn, 1e-3 * rand(plen))
  function dfunc(dstate, state, p, t)
    set_params(nn, p)
    scratch = zeros(eltype(dstate), 4)
    dstate[impart] .= -1.0 .*
                      (H_0 * state[repart] .+ cos(2.0 * t) .* H_D * state[repart])
    dstate[repart] .= H_0 * state[impart] .+ cos(2.0 * t) .* H_D * state[impart]
    applyNN!(nn, dstate, scratch)
    dstate .+= scratch
    nothing
  end
  return dfunc, nn
end

dfunc, nn = make_dfunc(Float64)
##cell initialize and solve
y0 = [1.0, 0.0, 0.0, 0.0]
p = get_params(nn)
tspan = (0, 20.0)

#test dfunc works
ds = zero(y0)
dfunc(ds, y0, p, 0.2) #test dfunc works

#get solution
prob = ODEProblem{true}(dfunc, y0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-10)
##cell
const target = zero(y0);
target[2] = 1.0;
function g(u, p, t)
  dot(u, target)^2
end

function gintegrate(p)
  dfunc, nn = make_dfunc(eltype(p))
  set_params(nn, p)
  prob = ODEProblem{true}(dfunc, y0, tspan, p)
  sol = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)
  integral, error = quadgk((t) -> (g(sol(t), p, t)), tspan...)
  return integral
end
refdp = ForwardDiff.gradient(gintegrate, p)


du4, dp4 = adjoint_sensitivities(sol, Tsit5(), g=g,
  sensealg=InterpolatingAdjoint(autodiff=true, autojacvec=EnzymeVJP()),
  abstol=1e-12, reltol=1e-12)
@test isapprox(dp4', refdp, atol=1e-5)

# du1, dp1 = adjoint_sensitivities(sol, Tsit5(), g=g,
#   sensealg=BacksolveAdjoint(autodiff=true, autojacvec=EnzymeVJP()),
#   abstol=1e-12, reltol=1e-12)
# @test isapprox(dp1', refdp, atol=1e-5)

# du2, dp2 = adjoint_sensitivities(
#   sol, Tsit5(), g=g, sensealg=GaussAdjoint(autodiff=true, autojacvec=EnzymeVJP()),
#   abstol=1e-12, reltol=1e-12)
# @test isapprox(dp2', refdp, atol=1e-5)

# du3, dp3 = adjoint_sensitivities(sol, Tsit5(), g=g,
#   sensealg=QuadratureAdjoint(autodiff=true, autojacvec=EnzymeVJP()),
#   abstol=1e-12, reltol=1e-12)
# @test isapprox(dp3', refdp, atol=1e-5)
