using Parameters: @with_kw


@with_kw mutable struct Soil{T<:AbstractFloat}
  n_layers::Int = 5                                 # 土壤层数
  K::Vector{T} = zeros(T, n_layers)                 # 节点水力传导度 [cm/s]
  K₊ₕ::Vector{T} = zeros(T, n_layers + 1)           # 界面水力传导度 [cm/s]
  ψ::Vector{T} = zeros(T, n_layers)                 # 土壤水势 [cm]
  θ_prev::Vector{T} = zeros(T, n_layers)            # 上一时刻含水量 [-]
  θ::Vector{T} = zeros(T, n_layers)                 # 当前含水量 [-]
  Q::Vector{T} = zeros(T, n_layers + 1)             # 达西通量 [cm/s]
end


# van Genuchten 水力传导度函数 K(θ)
function hydraulic_conductivity(θ, θ_s, θ_r, Ks, n)
  m = 1.0f0 - 1.0f0 / n
  Se = (θ - θ_r) / (θ_s - θ_r)  # 有效饱和度
  Se = clamp(Se, 0.01f0, 0.99f0)  # 避免极值
  K = Ks * sqrt(Se) * (1.0f0 - (1.0f0 - Se^(1.0f0 / m))^m)^2
  return K
end

# van Genuchten 土壤水势函数 ψ(θ)
function soil_water_potential(θ, θ_s, θ_r, α, n)
  m = 1.0f0 - 1.0f0 / n
  Se = (θ - θ_r) / (θ_s - θ_r)
  Se = clamp(Se, 0.01f0, 0.99f0)
  ψ = -1.0f0 / α * (Se^(-1.0f0 / m) - 1.0f0)^(1.0f0 / n)
  return ψ
end
