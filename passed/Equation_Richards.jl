using Parameters: @with_kw


@with_kw mutable struct Soil{T<:AbstractFloat}
  n_layers::Int = 5                                 # 土壤层数
  K::Vector{T} = zeros(T, n_layers)                 # 节点水力传导度 [cm/h]
  K₊ₕ::Vector{T} = zeros(T, n_layers + 1)           # 界面水力传导度 [cm/h]
  ψ::Vector{T} = zeros(T, n_layers)                 # 土壤水势 [cm]
  θ_prev::Vector{T} = zeros(T, n_layers)            # 上一时刻含水量 [-]
  θ::Vector{T} = zeros(T, n_layers)                 # 当前含水量 [-]
  Q::Vector{T} = zeros(T, n_layers + 1)             # 达西通量 [cm/h]
end



function cal_Q!(soil::Soil{T}, ps::StructVector{P}, θ::AbstractVector{T};
  θ0::T=NaN, Q0::T=NaN, method="θ0", CFL=0.5)::T where {T<:Real,P<:AbstractSoilParam{T}}

  (; ibeg, N, Q, Q_CFL, K, K₊ₕ, ψ) = soil
  Δz₊ₕ = soil.Δz₊ₕ_cm
  dt = soil.dt / 3600.0 # [s] -> [h]

  cal_K!(soil, ps, θ)
  cal_ψ!(soil, ps, θ)

  method == "θ0" && (Q0 = boundary_top(soil, ps, θ0; CFL)[1])

  # # 防止剧烈震荡
  # @inbounds for i in ibeg:N-1
  #   Q_CFL[i] = CFL * θ[i] * Δz₊ₕ[i] / dt # [cm h-1]
  # end
  # @show maximum(Q_CFL)
  # Qmax = median(@view Q_CFL[ibeg:N])

  @inbounds for i in ibeg:N-1
    _Q = -K₊ₕ[i] * ((ψ[i] - ψ[i+1]) / Δz₊ₕ[i] + 1.0) # [cm h-1]
    Q[i] = _Q #clamp(_Q, -Qmax, Qmax)
  end

  Q[N] = -K[N] # 尾部重力排水
  soil.Q0 = Q0 #clamp(Q0, -Qmax, Qmax)
  return Q0
end


"""
# Arguments
- `method`: 
  + `ψ0`: ψ0 boundary condition, 第一类边界条件
  + `Q0`: Q0 boundary condition, 第二类边界条件
> dθ: actually is dθ/dt
"""
function RichardsEquation(dθ::AbstractVector{T}, θ::AbstractVector{T}, ps::StructVector{P}, t;
  soil::Soil{T}, method="θ0") where {T<:Real,P<:AbstractSoilParam{T}}

  soil.timestep += 1
  # mod(p.timestep, 1000) == 0 && println("timestep = ", p.timestep)

  (; ibeg, N, Q, θ0, Q0, sink) = soil # Δz, z, 
  Δz = soil.Δz_cm
  Q0 = cal_Q!(soil, ps, θ; θ0, Q0, method)
  # @show θ0, Q0, method

  dθ[ibeg] = ((-Q0 + Q[ibeg]) - sink[ibeg]) / Δz[ibeg] / 3600.0
  @inbounds for i in ibeg+1:N
    dθ[i] = ((-Q[i-1] + Q[i]) - sink[i]) / Δz[i] / 3600.0 # [m3 m-3] / h-1
  end
end
