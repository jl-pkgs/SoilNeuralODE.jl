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
end

function van_genuchten_K(θ, p)
  θ_s, θ_r, Ks, α, n = p
  m = 1.0 - 1.0 / n
  Se = (θ - θ_r) / (θ_s - θ_r)
  Se = max(1e-5, min(1.0, Se))
  return Ks * sqrt(Se) * (1.0 - (1.0 - Se^(1.0 / m))^m)^2
end

function van_genuchten_psi(θ, p)
  θ_s, θ_r, Ks, α, n = p
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
    # Interface
    K_mid = (soil.K[i] + soil.K[i+1]) / 2.0
    # Flux
    soil.Q[i+1] = -K_mid * ((soil.ψ[i+1] - soil.ψ[i]) / soil.dz + 1.0)
  end
  
  # Bottom boundary: Free drainage
  soil.Q[end] = -soil.K[end]
end

# 2. ODE Function (Richards Equation)
function richards_eq_inner!(dθ, θ, p, t, soil)
  # p is just params
  
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
p_true = [0.45, 0.05, 10.0, 0.01, 2.0]
u0 = fill(0.2, 20)
tspan = (0.0, 1.0)

# Create Soil struct
soil_struct = Soil{Float64}(n_layers=length(u0))

# Initial dummy problem (will be remade)
prob = ODE.ODEProblem((dθ, θ, p, t) -> richards_eq_inner!(dθ, θ, p, t, soil_struct), u0, tspan, p_true)


# 4. Loss Function (Adjoint)
function loss_adjoint_enzyme(u0, p, soil, prob, alg=ODE.Tsit5())
  sensealg = SMS.InterpolatingAdjoint(autojacvec=SMS.EnzymeVJP())
  
  # Create a closure that captures the passed 'soil'
  # This ensures Enzyme tracks the specific 'soil' instance (and its shadow)
  f_closure = (dθ, θ, p, t) -> richards_eq_inner!(dθ, θ, p, t, soil)
  
  # Remake problem with new parameters and new function closure
  new_prob = ODE.remake(prob, f=f_closure, u0=u0, p=p)
  
  sol = ODE.solve(new_prob, alg; saveat=0.1, sensealg)
  return sum(sum(sol))
end


# 5. Differentiation
begin
  println("Starting Enzyme autodiff...")
  dp = make_zero(p_true)
  du0 = make_zero(u0)
  
  # Create shadow copy for soil struct
  dsoil = make_zero(soil_struct)

  # Enzyme.API.strictAliasing!(false) # Might be needed
  @time Enzyme.autodiff(
    Enzyme.Reverse,
    loss_adjoint_enzyme,
    Enzyme.Active,
    Enzyme.Duplicated(u0, du0),
    Enzyme.Duplicated(p_true, dp),
    Enzyme.Duplicated(soil_struct, dsoil),
    Enzyme.Const(prob),
    # Enzyme.Const(ODE.Rodas5P())
  )

  # Rodas5P(), QNDF(), FBDF()
  println("Gradient w.r.t u0: ", du0)
  println("Gradient w.r.t p:  ", dp)
end
