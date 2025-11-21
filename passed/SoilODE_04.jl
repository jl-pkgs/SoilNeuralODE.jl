import OrdinaryDiffEq as ODE
import SciMLSensitivity as SMS
import Enzyme
import Enzyme: make_zero
using ComponentArrays
using Parameters: @with_kw

# 1. Soil Struct and Physics
@with_kw mutable struct Soil{T<:AbstractFloat}
  n_layers::Int = 20
  dz::T = 1.0
  K::Vector{T} = zeros(T, n_layers)
  ψ::Vector{T} = zeros(T, n_layers)
  Q::Vector{T} = zeros(T, n_layers + 1)
  ps::ComponentVector{T,Vector{T}} = ComponentVector(θ_s=0.45, θ_r=0.05, Ks=10.0, α=0.01, n=2.0)
end

function van_genuchten_K(θ, p)
  θ_s, θ_r, Ks, α, n = p.θ_s, p.θ_r, p.Ks, p.α, p.n
  m = 1.0 - 1.0 / n
  Se = (θ - θ_r) / (θ_s - θ_r)
  Se = max(1e-5, min(1.0, Se))
  return Ks * sqrt(Se) * (1.0 - (1.0 - Se^(1.0 / m))^m)^2
end

function van_genuchten_psi(θ, p)
  θ_s, θ_r, Ks, α, n = p.θ_s, p.θ_r, p.Ks, p.α, p.n
  m = 1.0 - 1.0 / n
  Se = (θ - θ_r) / (θ_s - θ_r)
  Se = max(1e-5, min(1.0, Se))
  return -1.0 / α * (Se^(-1.0 / m) - 1.0)^(1.0 / n)
end

function update_soil_property!(soil::Soil, θ, p)
  @inbounds for i in 1:soil.n_layers
    soil.K[i] = van_genuchten_K(θ[i], p)
    soil.ψ[i] = van_genuchten_psi(θ[i], p)
  end
end

function cal_flux!(soil::Soil, p)
  # Boundary conditions
  q_top = -0.5 # Infiltration
  soil.Q[1] = q_top

  @inbounds for i in 1:soil.n_layers-1
    K_mid = (soil.K[i] + soil.K[i+1]) / 2.0 # Interface
    soil.Q[i+1] = -K_mid * ((soil.ψ[i+1] - soil.ψ[i]) / soil.dz + 1.0)
  end
  soil.Q[end] = -soil.K[end] # Bottom boundary: Free drainage
end


# 2. ODE Function (Richards Equation)
function richards_eq_inner!(dθ, θ, p_vec, t, soil)
  soil.ps .= p_vec
  p = soil.ps
  # p = ComponentArray(p_vec, p_axes)

  # Update state-dependent variables
  update_soil_property!(soil, θ, p)
  cal_flux!(soil, p)

  # Compute derivatives
  @inbounds for i in 1:soil.n_layers
    dθ[i] = -(soil.Q[i+1] - soil.Q[i]) / soil.dz
  end
  return nothing
end

# 3. Problem Setup
# Parameters: θ_s, θ_r, Ks, α, n
p_true = ComponentArray(θ_s=0.45, θ_r=0.05, Ks=10.0, α=0.01, n=2.0)
p_flat = collect(p_true)
u0 = fill(0.2, 20)
tspan = (0.0, 1.0)
soil = Soil{Float64}(n_layers=length(u0))

# Initial dummy problem (will be remade)
prob = ODE.ODEProblem((dθ, θ, p, t) -> richards_eq_inner!(dθ, θ, p, t, soil), u0, tspan, p_flat)

# 4. Loss Function (Adjoint)
function loss_adjoint_enzyme(u0, p_vec, soil, prob, alg=ODE.Tsit5())
  sensealg = SMS.InterpolatingAdjoint(autojacvec=SMS.EnzymeVJP())
  f_closure(dθ, θ, p, t) = richards_eq_inner!(dθ, θ, p, t, soil)

  # Remake problem with new parameters and new function closure
  new_prob = ODE.remake(prob, f=f_closure, u0 = u0, p=p_vec)

  sol = ODE.solve(new_prob, alg; saveat=0.1, sensealg=sensealg)
  return sum(sum(sol))
end


# 5. Differentiation
begin
  println("Starting Enzyme autodiff...")
  dp_flat = make_zero(p_flat)
  du0 = make_zero(u0)

  # Create shadow copy for soil struct
  dsoil = make_zero(soil)

  # Enzyme.API.strictAliasing!(false) # Might be needed
  @time Enzyme.autodiff(
    Enzyme.Reverse,
    loss_adjoint_enzyme,
    Enzyme.Active,
    Enzyme.Duplicated(u0, du0),
    Enzyme.Duplicated(p_flat, dp_flat),
    Enzyme.Duplicated(soil, dsoil),
    Enzyme.Const(prob)
  )

  println("Gradient p: ", dp_flat)
  println("Gradient u0: ", du0)
end
