import OrdinaryDiffEq as ODE
import SciMLSensitivity as SMS
import Enzyme
import Enzyme: make_zero
using ComponentArrays, OffsetArrays
using Optimisers
using Parameters: @with_kw
using SoilNeuralODE: soil_depth_init, cal_Δz
using RTableTools, DataFrames
using Random

# # # Plotting
# using Plots
# gr(framestyle=:box)
# function plot_layer(i)
#   plot(θ_obs[i, :], label="Observed", title="Soil Layer $(i)", xlabel="Time Step", ylabel="Soil Moisture θ")
#   plot!(ysim_opt[i, :], label="Optimized", lw=2)
#   # plot!(ysim[i, :], label="Initial", ls=:dash)
# end


## 其中观测数据在[5, 10, 20, 50, 100]，也即是: 
function load_soil_data(f)
  df = fread(f)
  # 提取土壤水分观测 (5层: 5, 10, 20, 50, 100 cm)
  θ_obs = Matrix(df[:, [:SOIL_MOISTURE_5, :SOIL_MOISTURE_10, :SOIL_MOISTURE_20,
    :SOIL_MOISTURE_50, :SOIL_MOISTURE_100]])
  # 提取降水 [mm] -> [cm]
  P = Float32.(df.P_CALC ./ 10.0)
  return θ_obs', P  # 转置为 (n_layers, n_times)
end


# 1. Soil Struct and Physics
@with_kw mutable struct Soil{T<:AbstractFloat}
  n_layers::Int = 20
  z::OffsetVector{T} = zeros(T, n_layers + 2)
  Δz₊ₕ::Vector{T} = zeros(T, n_layers)
  # dz::T = 1.0
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

function cal_flux!(soil::Soil, p, t)
  # Boundary conditions
  q_top = -0.0 # Infiltration, [cm h-1], 5mm h-1
  soil.Q[1] = q_top

  @inbounds for i in 1:soil.n_layers-1
    dz = i == 1 ? abs(soil.z[1]) : soil.Δz₊ₕ[i]
    dz = dz * 100 # [m] to [cm], z[i] - z[i+1]
    K_mid = (soil.K[i] + soil.K[i+1]) / 2.0 # Interface
    soil.Q[i+1] = -K_mid * ((soil.ψ[i] - soil.ψ[i+1]) / dz + 1.0)
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
  cal_flux!(soil, p, t)

  # Compute derivatives
  @inbounds for i in 1:soil.n_layers
    dz = i == 1 ? abs(soil.z[1]) : soil.Δz₊ₕ[i]
    dz = dz * 100 # [m] to [cm], z[i] - z[i+1]
    dθ[i] = -(soil.Q[i] - soil.Q[i+1]) / dz # dθ/dt
  end
  return nothing
end


# Rodas5P, Tsit5
# 5. Prediction Function
function predict(u0, p_vec, soil, prob, tspan; saveat=0.5, alg=ODE.Tsit5())
  sensealg = SMS.InterpolatingAdjoint(autojacvec=SMS.EnzymeVJP())

  f_closure(dθ, θ, p, t) = richards_eq_inner!(dθ, θ, p, t, soil)
  new_prob = ODE.remake(prob, f=f_closure, u0=u0, p=p_vec, tspan=tspan)
  sol = ODE.solve(new_prob, alg; saveat, sensealg, maxiters=1e5)
  return Array(sol)
end

# 4. Loss Function (Adjoint)
function loss_adjoint_enzyme(u0, p_vec, soil, yobs, prob, tspan, alg=ODE.Tsit5())
  # ysim includes t=0, so we take 2:end to match yobs (t=1...ntime)
  ysim = predict(u0, p_vec, soil, prob, tspan; saveat=1, alg)[inds, :]

  loss = (yobs - ysim) .^ 2 |> sum # SSE
  @show loss
  return loss
  # return sum(ysim)
end


# 6. Optimization
function train(p_init, u0, soil, yobs, prob, tspan; epochs=50, lr=0.05, alg=ODE.Rodas5P())
  p = copy(p_init)
  opt = Optimisers.Adam(lr)
  opt_state = Optimisers.setup(opt, p)

  dp = make_zero(p)
  du0 = make_zero(u0)
  dsoil = make_zero(soil)

  for i in 1:epochs
    fill!(dp, 0.0)
    fill!(du0, 0.0)

    # Forward & Backward
    # Note: loss_adjoint_enzyme prints the loss
    Enzyme.autodiff(
      Enzyme.Reverse,
      loss_adjoint_enzyme,
      Enzyme.Active,
      Enzyme.Duplicated(u0, du0),
      Enzyme.Duplicated(p, dp),
      Enzyme.Duplicated(soil, dsoil),
      Enzyme.Const(yobs),
      Enzyme.Const(prob),
      Enzyme.Const(tspan),
      Enzyme.Const(alg)
    )
    opt_state, p = Optimisers.update(opt_state, p, dp)

    # Constraints (Ks > 0, etc.)
    p .= max.(p, 1e-5)
    @show p

    if i % 10 == 0
      println("Epoch $i | Gradients: ", dp)
    end
  end
  return p
end


# ## 第一个时刻进行初始化
# begin
#   using Plots

#   function plot_layer(i)
#     plot(θ_obs[i, :], title="Soil Layer $(i)", xlabel="Time Step", ylabel="Soil Moisture θ")
#     plot!(ysim[i, :], label="Prediction")
#   end

#   ps = map(plot_layer, 1:5)
#   p = plot(ps..., layout=(3, 2), size=(800, 600))
#   savefig(p, "soil_moisture_prediction.png")
# end
