export Retention, Retention_K, Retention_θ, Retention_ψ, Retention_∂θ∂ψ, Retention_∂K∂Se
export Retention_∂K∂θ, Retention_∂ψ∂θ
export Ponding


using Parameters, StructArrays

import FieldMetadata: @metadata, @units, units
@metadata bounds nothing

abstract type AbstractSoilParam{T<:Real} end


@bounds @with_kw mutable struct VanGenuchten{T<:Real} <: AbstractSoilParam{T}
  θ_sat::T = 0.287 | (0.2, 0.5)   # [m3 m-3]
  θ_res::T = 0.075 | (0.03, 0.2)  # [m3 m-3]
  Ksat::T = 34.0 | (0.002, 60.0)  # [cm h-1]
  α::T = 0.027 | (0.002, 0.300)   # [cm-1]
  n::T = 3.96 | (1.05, 4.0)       # [-]
  m::T = 1.0 - 1.0 / n    # ! not optimize, [-]
  ψ_sat::T = -10.0        # ! not optimize, [cm]
end

@bounds @with_kw mutable struct Campbell{T<:Real} <: AbstractSoilParam{T}
  θ_sat::T = 0.287 | (0.2, 0.5)     # [m3 m-3]
  ψ_sat::T = -10.0 | (-100.0, -5.0) # [cm]
  Ksat::T = 34.0 | (0.002, 100.0)   # [cm h-1]
  b::T = 4.0 | (3.0, 15.0)          # [-]
end

@bounds @with_kw mutable struct Lateral{T<:Real} <: AbstractSoilParam{T}
  α_drain::T = 0.1 | (0.0, 10.0)  # drainage rate coefficient, α_drain = S_0 / l_eff
  θ_lat::T = 0.15 | (0.03, 0.3)   # [m3 m-3], 与θ_res类似，但为计算q_lat，超过该值才有侧向流
  γ::T = 1.0 | (0.1, 2.0)         # [-], lateral flow exponent, J = Se^γ S_0
end


"""
c_inf = 1.0 / n * sqrt(S0) * 3600 * 10000.0 / (100L) # [cm^(1/3) h-1]
drainage = c_inf * h_m^(5 / 3) # [1 cm h-1]
"""
@bounds @with_kw mutable struct Ponding{FT}
  h_pond_min::FT = 0.1 | (0.05, 1.00) # [cm], 地表自由水的蓄水容量, 低于该数值不产流
  h_pond_max::FT = 2.0 | (1.00, 5.00) # [cm], 高于高数值全部产流
  c_inf::FT = 1000.0 | (50.0, 3000.0) #
end


"""
# Example: 
- S_0 = 0.1 m m-1, l = 10m = 10,00cm, α_drain = S_0 / l = 1e-4
- S_0 = 0.1 m m-1, l = 1m = 1,00cm, α_drain = S_0 / l = 1e-3
"""
@bounds @with_kw mutable struct SoilHydra{T<:AbstractFloat,P<:AbstractSoilParam{T}}
  ## Ponding and Infiltration
  ponding::Ponding{T} = Ponding{T}()

  ## 侧向壤中流
  # α_drain::T = 0.1 | (0.0, 10.0)  # drainage rate coefficient, α_drain = S_0 / l_eff
  use_lateral::Bool = false
  lateral::StructVector{Lateral{T}}

  ## Richards方程参数
  hydraulic::StructVector{P}

  ## Soil thermal properties
  # thermal::SoilThermal{FT} = SoilThermal{FT}()
end


# SoilHydra
function SoilHydra(p::P, N::Int; kw...) where {T<:Real,P<:AbstractSoilParam{T}}
  lateral = StructVector([Lateral{T}() for _ in 1:N])
  hydraulic = StructVector([deepcopy(p) for _ in 1:N])
  SoilHydra{T,P}(; lateral, hydraulic, kw...)
end


# 仅用于计算侧向壤中流
function Retention_Se(θ::T, θ_lat::T, par::Campbell{T}) where {T<:Real}
  se = (θ - θ_lat) / (par.θ_sat)
  clamp(se, T(0.0), T(1.0))
end

function Retention_Se(θ::T, θ_lat::T, par::VanGenuchten{T}) where {T<:Real}
  se = (θ - θ_lat) / (par.θ_sat - par.θ_res)
  clamp(se, T(0.0), T(1.0))
end


# 多重派发(runtime-dispatch)可能会导致速度变慢
Retention(ψ::T, par::Campbell{T}) where {T<:Real} = Campbell(ψ, par)
Retention_K(θ::T, par::Campbell{T}) where {T<:Real} = Campbell_K(θ, par)
Retention_θ(ψ::T, par::Campbell{T}) where {T<:Real} = Campbell_θ(ψ, par)
Retention_ψ(θ::T, par::Campbell{T}) where {T<:Real} = Campbell_ψ(θ, par)
Retention_ψ_Se(Se::T, par::Campbell{T}) where {T<:Real} = Campbell_ψ_Se(Se, par)
Retention_∂K∂θ(θ::T, par::Campbell{T}) where {T<:Real} = Campbell_∂K∂θ(θ, par)
Retention_∂θ∂ψ(ψ::T, par::Campbell{T}) where {T<:Real} = Campbell_∂θ∂ψ(ψ, par)
Retention_∂ψ∂θ(ψ::T, par::Campbell{T}) where {T<:Real} = Campbell_∂ψ∂θ(ψ, par)
Retention_∂K∂Se(Se::T, par::Campbell{T}) where {T<:Real} = Campbell_∂K∂Se(Se, par)


Retention(ψ::T, par::VanGenuchten{T}) where {T<:Real} = van_Genuchten(ψ, par)
Retention_θ(ψ::T, par::VanGenuchten{T}) where {T<:Real} = van_Genuchten_θ(ψ, par)
Retention_K(θ::T, par::VanGenuchten{T}) where {T<:Real} = van_Genuchten_K(θ, par)
Retention_ψ(θ::T, par::VanGenuchten{T}) where {T<:Real} = van_Genuchten_ψ(θ, par)
Retention_ψ_Se(Se::T, par::VanGenuchten{T}) where {T<:Real} = van_Genuchten_ψ_Se(Se, par)
Retention_∂θ∂ψ(ψ::T, par::VanGenuchten{T}) where {T<:Real} = van_Genuchten_∂θ∂ψ(ψ, par)
Retention_∂ψ∂θ(ψ::T, par::VanGenuchten{T}) where {T<:Real} = van_Genuchten_∂ψ∂θ(ψ, par)
Retention_∂K∂Se(Se::T, par::VanGenuchten{T}) where {T<:Real} = van_Genuchten_∂K∂Se(Se, par)
Retention_∂K∂θ(θ::T, par::VanGenuchten{T}) where {T<:Real} = van_Genuchten_∂K∂θ(θ, par)


Retention(ψ::T; par::AbstractSoilParam{T}) where {T<:Real} = Retention(ψ, par)
Retention_K(θ::T; par::AbstractSoilParam{T}) where {T<:Real} = Retention_K(θ, par)
Retention_θ(ψ::T; par::AbstractSoilParam{T}) where {T<:Real} = Retention_θ(ψ, par)
Retention_ψ(θ::T; par::AbstractSoilParam{T}) where {T<:Real} = Retention_ψ(θ, par)
Retention_ψ_Se(Se::T; par::AbstractSoilParam{T}) where {T<:Real} = Retention_ψ_Se(Se, par)
Retention_∂θ∂ψ(ψ::T; par::AbstractSoilParam{T}) where {T<:Real} = Retention_∂θ∂ψ(ψ, par)
Retention_∂ψ∂θ(ψ::T; par::AbstractSoilParam{T}) where {T<:Real} = Retention_∂ψ∂θ(ψ, par)
Retention_∂K∂Se(Se::T; par::AbstractSoilParam{T}) where {T<:Real} = Retention_∂K∂Se(Se, par)


# # 这里把侧向出流加入到sink中
# function update_sink!(soil::Soil, param::SoilHydra)
#   !param.use_lateral && return # 没有侧向流则直接返回

#   # (; α_drain) = param
#   (; α_drain, γ, θ_lat) = param.lateral
#   ps = param.hydraulic

#   (; sink, ET, q_lat, θ, Δz, N, ibeg, K) = soil
#   for i in ibeg:N
#     Se = Retention_Se(θ[i], θ_lat[i], ps[i])
#     Δz_cm = Δz[i] * 100
#     q_lat[i] = K[i] * Se^γ[i] * Δz_cm * α_drain[i] # α_drain = S_0 / l, l in cm 
#     sink[i] = ET[i] + q_lat[i]
#   end
# end

# function cal_θ!(soil::Soil{T}, ps::StructVector{P}, ψ::AbstractVector{T}) where {T<:Real,P<:AbstractSoilParam{T}}
#   (; N, ibeg, θ) = soil
#   @inbounds for i = ibeg:N
#     θ[i] = Retention_θ(ψ[i], ps[i])
#   end
# end

# function cal_∂θ∂ψ!(soil::Soil{T}, ps::StructVector{P}, ψ::AbstractVector{T}) where {T<:Real,P<:AbstractSoilParam{T}}
#   (; ibeg, N, ∂θ∂ψ) = soil
#   @inbounds for i in ibeg:N
#     ∂θ∂ψ[i] = Retention_∂θ∂ψ(ψ[i], ps[i])
#   end
# end

# export cal_∂θ∂ψ!

# # 算术平均、调和平均
# mean_arithmetic(K1::T, K2::T, d1::T, d2::T) where {T<:Real} = (K1 * d1 + K2 * d2) / (d1 + d2)
# mean_harmonic(K1::T, K2::T, d1::T, d2::T) where {T<:Real} = K1 * K2 * (d1 + d2) / (K1 * d2 + K2 * d1)

# # cal_ψ!(soil::Soil) = cal_ψ!(soil, soil.θ)
# function cal_ψ!(soil::Soil{T}, ps::StructVector{P}, θ::AbstractVector{T}) where {T<:Real,P<:AbstractSoilParam{T}}
#   (; N, ψ) = soil
#   i0 = max(soil.ibeg - 1, 1)

#   @inbounds for i = i0:N
#     ψ[i] = Retention_ψ(θ[i], ps[i])
#   end
# end

# # 一并更新K和K₊ₕ
# # cal_K!(soil::Soil) = cal_K!(soil, soil.θ)

# function cal_K!(soil::Soil{T}, ps::StructVector{P}, θ::AbstractVector{T}) where {T<:Real,P<:AbstractSoilParam{T}}
#   (; N, K, K₊ₕ, Δz) = soil
#   # param = soil.param.param
#   i0 = max(soil.ibeg - 1, 1)

#   @inbounds for i = i0:N
#     K[i] = Retention_K(θ[i], ps[i])
#   end
#   @inbounds for i = i0:N-1
#     K₊ₕ[i] = mean_arithmetic(K[i], K[i+1], Δz[i], Δz[i+1])
#     # K₊ₕ[i] = mean_harmonic(K[i], K[i+1], Δz[i], Δz[i+1])
#   end
#   K₊ₕ[N] = K[N]
# end


export specific_yield!
export cal_K!, cal_ψ!, cal_θKCap!, cal_K₊ₕ!
export Init_SoilWaterParam, Init_ψ0
