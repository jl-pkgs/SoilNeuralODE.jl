import OrdinaryDiffEq as ODE
import SciMLSensitivity as SMS
import Enzyme
import Enzyme: make_zero
using ComponentArrays

# 1. Physics Functions (Van Genuchten)
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

# 2. ODE Function (Richards Equation)
# Simplified version of Equation_Richards.jl
function richards_eq!(dθ, θ, p, t)
  # p = [θ_s, θ_r, Ks, α, n]
  dz = 1.0
  N = length(θ)

  # Boundary conditions
  q_top = -0.5 # Infiltration
  q_in = q_top

  @inbounds for i in 1:N
    # Current cell
    K_i = van_genuchten_K(θ[i], p)
    psi_i = van_genuchten_psi(θ[i], p)

    if i < N
      # Next cell
      K_next = van_genuchten_K(θ[i+1], p)
      psi_next = van_genuchten_psi(θ[i+1], p)

      # Interface
      K_mid = (K_i + K_next) / 2.0

      # Flux
      q_out = -K_mid * ((psi_next - psi_i) / dz + 1.0)
    else
      # Bottom boundary: Free drainage
      q_out = -K_i
    end

    dθ[i] = -(q_out - q_in) / dz
    q_in = q_out
  end
  return nothing
end

# 3. Problem Setup
# Parameters: θ_s, θ_r, Ks, α, n
p_true = [0.45, 0.05, 10.0, 0.01, 2.0]
u0 = fill(0.2, 20)
tspan = (0.0, 1.0)

prob = ODE.ODEProblem(richards_eq!, u0, tspan, p_true)

# 4. Loss Function (Adjoint)
function loss_adjoint_enzyme(u0, p, prob, alg=ODE.Tsit5())
  sensealg = SMS.InterpolatingAdjoint(autojacvec=SMS.EnzymeVJP())
  new_prob = ODE.remake(prob, u0=u0, p=p)
  sol = ODE.solve(new_prob, alg; saveat=0.1, sensealg)
  return sum(sum(sol))
end


# 5. Differentiation
begin
  println("Starting Enzyme autodiff...")
  dp = make_zero(p_true)
  du0 = make_zero(u0)

  # Enzyme.API.strictAliasing!(false) # Might be needed
  @time Enzyme.autodiff(
    Enzyme.Reverse,
    loss_adjoint_enzyme,
    Enzyme.Active,
    Enzyme.Duplicated(u0, du0),
    Enzyme.Duplicated(p_true, dp),
    Enzyme.Const(prob),
    # Enzyme.Const(ODE.Rodas5P())
  )

  # Rodas5P(), QNDF(), FBDF()
  println("Gradient w.r.t u0: ", du0)
  println("Gradient w.r.t p:  ", dp)
end
