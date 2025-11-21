"""
    van_Genuchten(ψ::T, par::VanGenuchten{T})

van Genuchten (1980) relationships

# Arguments
+ `ψ`: Matric potential
+ `param`
  - `θ_res`       : Residual water content
  - `θ_sat`       : Volumetric water content at saturation
  - `Ksat`        : Hydraulic conductivity at saturation (cm/s)
  - `α`           : Inverse of the air entry potential (cm-1)
  - `n`           : Pore-size distribution index
  - `m`           : Exponent
  - `soil_texture`: Soil texture flag

# Examples
```julia
# Haverkamp et al. (1977): sand
param = (soil_texture = 1, 
  θ_res = 0.075, θ_sat = 0.287, 
  α = 0.027, n = 3.96, m = 1, Ksat = 34 / 3600)

# Haverkamp et al. (1977): Yolo light clay
param = (soil_texture=2, 
  θ_res = 0.124, θ_sat = 0.495,
  α = 0.026, n = 1.43, m = 1 - 1 / 1.43,
  Ksat = 0.0443 / 3600)
```
"""
@inline function van_Genuchten(ψ::T, par::VanGenuchten{T}) where {T<:Real}
  θ = van_Genuchten_θ(ψ, par)
  K = van_Genuchten_K(θ, par)
  ∂θ∂ψ = van_Genuchten_∂θ∂ψ(ψ, par)
  θ, K, ∂θ∂ψ
end

# @fastmath 
function van_Genuchten_θ(ψ::T, par::VanGenuchten{T}; ψ_min::T=T(-1e7)) where {T<:Real}
  (; θ_res, θ_sat, α, n, m) = par
  ψ <= ψ_min && return θ_res

  # Effective saturation (Se) for specified matric potential (ψ)
  Se = ψ <= 0 ? (1 + (α * abs(ψ))^n)^-m : 1.0
  Se = clamp(Se, T(0.0), T(1.0))

  # Volumetric soil moisture (θ) for specified matric potential (ψ)
  return θ_res + (θ_sat - θ_res) * Se # θ
end


@inline function soft_clamp(x::T, low::T, high::T, k::T=T(20)) where T
  mid = (high + low) / 2
  half = (high - low) / 2
  return mid + half * tanh((x - mid) * k / half)
end

# 平滑的 ReLU (Softplus)，用于确保数值非负：log(1 + exp(k*x)) / k
@inline soft_plus(x::T, k::T=T(50)) where T = log1p(exp(x * k)) / k


# @fastmath
function van_Genuchten_K(θ::T, par::VanGenuchten{T}) where {T<:Real}
  (; θ_res, θ_sat, Ksat, m) = par

  # 1. 归一化并软截断 Se 到 [1e-6, 1.0]，避免 sqrt(0) 导数爆炸
  x = (θ - θ_res) / (θ_sat - θ_res)
  Se = soft_clamp(x, T(1e-6), T(1.0))

  # 2. 计算 K，使用 max 确保幂底数非负
  m_inv = 1 / m
  term = max(1 - Se^m_inv, zero(T))
  return Ksat * sqrt(Se) * (1 - term^m)^2
end

# @fastmath
function van_Genuchten_ψ(θ::T, par::VanGenuchten{T}; ψ_min=T(-1e7)) where {T<:Real}
  (; θ_res, θ_sat, α, n, m) = par

  # 1. 软截断 Se，防止除以 0
  x = (θ - θ_res) / (θ_sat - θ_res)
  Se = soft_clamp(x, T(1e-6), T(1.0))

  # 2. 计算 ψ
  # 使用 soft_plus 确保 (Se^(-1/m) - 1) > 0，防止复数域错误
  inv_Se_m = (1 / Se)^(1 / m)
  base = soft_plus(inv_Se_m - 1)
  ψ = -1 / α * base^(1 / n)
  # 3. 确保不低于 ψ_min (此处用简单 max 通常足以维持稳定，也可换 soft_max)
  return max(ψ, ψ_min)
end



function van_Genuchten_ψ_Se(Se::T, par::VanGenuchten{T}; ψ_min=T(-1e7)) where {T<:Real}
  (; α, n, m) = par
  Se <= 0.0 && return ψ_min
  Se >= 1.0 && return T(0.0)
  ψ = -1 / α * pow(pow(1.0 / Se, (1 / m)) - 1, 1 / n)
  return max(ψ, ψ_min) # Ensure the returned value does not go below ψ_min
end

# @fastmath 
function van_Genuchten_∂θ∂ψ(ψ::T, par::VanGenuchten{T})::T where {T<:Real}
  (; θ_res, θ_sat, α, n, m) = par
  if ψ <= 0.0
    num = α * m * n * (θ_sat - θ_res) * (α * abs(ψ))^(n - 1)
    den = (1 + (α * abs(ψ))^n)^(m + 1)
    ∂θ∂ψ = num / den
  else
    ∂θ∂ψ = T(0.0)
  end
  return ∂θ∂ψ
end

van_Genuchten_∂ψ∂θ(ψ::T, par::VanGenuchten{T}) where {T<:Real} = T(1.0) / van_Genuchten_∂θ∂ψ(ψ, par)

@inline function van_Genuchten_∂K∂Se(Se::T, par::VanGenuchten{T}) where {T<:Real}
  (; Ksat, m) = par
  f = 1 - (1 - Se^(1 / m))^m
  term1 = f^2 / (2 * sqrt(Se))
  term2 = 2 * Se^(1 / m - 1 / 2) * f / ((1 - Se^(1 / m))^(1 - m))
  return Ksat * (term1 + term2)
end

@inline function van_Genuchten_∂K∂θ(θ::T, par::VanGenuchten{T}) where {T<:Real}
  (; θ_res, θ_sat) = par
  Se = (θ - θ_res) / (θ_sat - θ_res)
  return van_Genuchten_∂K∂Se(Se, par) / (θ_sat - θ_res)
end


export van_Genuchten, van_Genuchten_θ, van_Genuchten_K, van_Genuchten_ψ,
  van_Genuchten_∂θ∂ψ, van_Genuchten_∂ψ∂θ, van_Genuchten_∂K∂Se,
  van_Genuchten_ψ_Se, 
  van_Genuchten_∂K∂θ

# Special case for:
# - `soil_type = 1`: Haverkamp et al. (1977) sand
# - `soil_type = 2`: Yolo light clay

# if soil_type == 1
#   K = Ksat * 1.175e6 / (1.175e6 + abs(ψ)^4.74)
# elseif soil_type == 2
#   K = Ksat * 124.6 / (124.6 + abs(ψ)^1.77)
# end
