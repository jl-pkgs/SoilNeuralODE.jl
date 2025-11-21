import OrdinaryDiffEq as ODE
import SciMLSensitivity as SMS
import Enzyme
import Enzyme: make_zero
using ComponentArrays, OffsetArrays
using Parameters: @with_kw
using SoilNeuralODE: soil_depth_init, cal_Δz

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


# 5. Prediction Function
function predict(u0, p_vec, soil, prob, tspan; saveat=0.5, alg=ODE.Tsit5())
  sensealg = SMS.InterpolatingAdjoint(autojacvec=SMS.EnzymeVJP())

  f_closure(dθ, θ, p, t) = richards_eq_inner!(dθ, θ, p, t, soil)
  new_prob = ODE.remake(prob, f=f_closure, u0=u0, p=p_vec, tspan=tspan)
  sol = ODE.solve(new_prob, alg; saveat, sensealg)
  return Array(sol)
end

# 4. Loss Function (Adjoint)
function loss_adjoint_enzyme(u0, p_vec, soil, yobs, prob, alg=ODE.Tsit5())
  ysim = predict(u0, p_vec, soil, prob, tspan; saveat=1, alg=alg)[inds, :]
  loss = (yobs - ysim) .^ 2 |> sum # SSE
  @show loss
  return loss
  # return sum(ysim)
end

## 其中观测数据在[5, 10, 20, 50, 100]，也即是: 

using RTableTools, DataFrames
function load_soil_data(f)
  df = fread(f)
  # 提取土壤水分观测 (5层: 5, 10, 20, 50, 100 cm)
  θ_obs = Matrix(df[:, [:SOIL_MOISTURE_5, :SOIL_MOISTURE_10, :SOIL_MOISTURE_20,
    :SOIL_MOISTURE_50, :SOIL_MOISTURE_100]])
  # 提取降水 [mm] -> [cm]
  P = Float32.(df.P_CALC ./ 10.0)
  return θ_obs', P  # 转置为 (n_layers, n_times)
end

data_file = "data/SM_AR_Batesville_8_WNW_2024.csv"
θ_obs, P = load_soil_data(data_file)
inds = [2, 3, 4, 5, 6]
ntime = size(θ_obs, 2) - 1

yobs = θ_obs[:, 2:end]  # 去掉第一个时刻的观测值
Q_t0 = θ_obs[:, 1]

begin
  # 供7层土壤
  z = -[1.25, 5, 10, 20, 50, 100.0, 200] ./ 100 # 第一层是虚拟的
  Δz = cal_Δz(z)
  z, z₋ₕ, z₊ₕ, Δz₊ₕ = soil_depth_init(Δz)
  n_layers = length(Δz)
  soil = Soil{Float64}(; n_layers, z, Δz₊ₕ)

  # 3. Problem Setup
  # Parameters: θ_s, θ_r, Ks, α, n
  p_true = ComponentArray(θ_s=0.45, θ_r=0.05, Ks=10.0, α=0.01, n=2.0)
  p_flat = collect(p_true)
  u0 = fill(0.2, n_layers)
  u0[inds] .= Q_t0
  u0[inds[end]+1:end] .= Q_t0[end]

  tspan = (1, ntime)

  # Initial dummy problem (will be remade)
  prob = ODE.ODEProblem((dθ, θ, p, t) -> richards_eq_inner!(dθ, θ, p, t, soil), u0, tspan, p_flat)
  @time ysim = predict(u0, p_flat, soil, prob, tspan; saveat=1)[inds, :]
end


## 第一个时刻进行初始化
begin
  using Plots

  function plot_layer(i)
    plot(θ_obs[i, :], title="Soil Layer $(i)", xlabel="Time Step", ylabel="Soil Moisture θ")
    plot!(ysim[i, :], label="Prediction")
  end

  ps = map(plot_layer, 1:5)
  p = plot(ps..., layout=(3, 2), size=(800, 600))
  savefig(p, "soil_moisture_prediction.png")
end


# 6. Differentiation & Prediction
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
    # Enzyme.Const(u0),
    Enzyme.Duplicated(p_flat, dp_flat),
    Enzyme.Duplicated(soil, dsoil),
    Enzyme.Const(yobs),
    Enzyme.Const(prob)
  )
  println("Gradient p: ", dp_flat)
  println("Gradient u0: ", du0)
end


