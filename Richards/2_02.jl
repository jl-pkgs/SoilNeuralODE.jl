using Lux, Enzyme, ComponentArrays, Optimisers, Random
using Parameters  # 用于 @with_kw 宏
using Statistics
using LinearAlgebra: norm


# 定义土壤状态结构体
@with_kw mutable struct Soil{T<:AbstractFloat}
  n_layers::Int                                     # 土壤层数
  K::Vector{T} = zeros(T, n_layers)                 # 节点水力传导度 [cm/s]
  K₊ₕ::Vector{T} = zeros(T, n_layers + 1)   # 界面水力传导度 [cm/s]
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

#=============================================================================
第二步(改进版):使用 Soil 结构体的 Richards 方程
=============================================================================#
function richards_explicit_step!(soil::Soil, soil_params, Δt, Δz)
  θ_s, θ_r, Ks, α, n = soil_params
  n_layers = length(soil.θ_prev)

  # 保存当前状态为前一时刻
  soil.θ_prev .= soil.θ

  # 计算每层的水力性质（节点值）
  for i in 1:n_layers
    soil.K[i] = hydraulic_conductivity(soil.θ_prev[i], θ_s, θ_r, Ks, n)
    soil.ψ[i] = soil_water_potential(soil.θ_prev[i], θ_s, θ_r, α, n)
  end

  # 计算界面水力传导度 (n+1 个界面)
  # 界面 1: 顶部边界
  soil.K₊ₕ[1] = soil.K[1]  # 或者设为 0
  for i in 2:n_layers
    soil.K₊ₕ[i] = 0.5f0 * (soil.K[i-1] + soil.K[i])
  end
  soil.K₊ₕ[end] = soil.K[end]  # 或者设为 0

  # 计算界面通量 (n+1 个界面)
  for i in 2:n_layers
    # 达西通量(向下为正)，使用存储的界面传导度
    soil.Q[i] = -soil.K₊ₕ[i] * ((soil.ψ[i] - soil.ψ[i-1]) / Δz + 1.0f0)
  end

  # 边界通量
  soil.Q[1] = 0.0f0      # 顶部无通量
  soil.Q[end] = 0.0f0    # 底部无通量

  # 更新含水量(通量散度)
  for i in 2:n_layers-1
    soil.θ[i] = soil.θ_prev[i] + Δt / Δz * (soil.Q[i] - soil.Q[i+1])
  end

  # 边界条件:顶层和底层保持不变(简化处理)
  soil.θ[1] = soil.θ_prev[1]
  soil.θ[end] = soil.θ_prev[end]

  return nothing
end

#=============================================================================
第三步:定义混合模型 = Richards物理模型 + 神经网络修正
=============================================================================#

# 神经网络:输入是深度,输出是对Richards模型的修正项
nn_model = Chain(Dense(1 => 16, tanh), Dense(16 => 1))

# 完整的前向模型：使用 Soil 结构体
function hybrid_forward(soil::Soil, depths, nn_params, st, soil_params, Δt, Δz)
  # 1. 物理模型预测（原地修改）
  richards_explicit_step!(soil, soil_params, Δt, Δz)

  # 2. 神经网络修正(基于深度)
  corrections, _ = nn_model(depths', nn_params, st)  # depths: (n_layers,) -> (1, n_layers)
  corrections = vec(corrections)  # (n_layers,)
  # @show corrections # 修正的是1个时刻, [n_layers, 1] => [n_layers, 16] => [n_layers, 1]

  # 3. 混合输出:physics + NN修正
  θ_hybrid = soil.θ .+ 0.01f0 .* corrections  # 小系数防止NN主导

  # 约束在合理范围
  θ_s, θ_r = soil_params[1], soil_params[2]
  θ_hybrid = clamp.(θ_hybrid, θ_r + 0.01f0, θ_s - 0.01f0)

  return θ_hybrid
end

#=============================================================================
第四步:损失函数(物理约束 + 数据拟合)
=============================================================================#
function loss_func(nn_params, soil, depths, θ_obs, nn_model, st, soil_params, Δt, Δz)
  # 前向预测（使用 Soil 结构体）
  θ_pred = hybrid_forward(soil, depths, nn_params, st, soil_params, Δt, Δz)

  # 数据拟合损失
  loss_data = mean((θ_pred .- θ_obs) .^ 2)

  # 物理约束:含水量不能超出饱和度
  θ_s = soil_params[1]
  loss_physics = sum(max.(0.0f0, θ_pred .- θ_s) .^ 2)

  return loss_data + 0.1f0 * loss_physics
end

#=============================================================================
第五步:测试梯度计算
=============================================================================#
begin
  rng = Xoshiro(123)

  # 土壤参数(典型壤土)
  θ_s = 0.45f0    # 饱和含水量
  θ_r = 0.05f0    # 残余含水量
  Ks = 0.01f0     # 饱和导水率 cm/s
  α = 0.02f0      # cm^-1
  n = 1.5f0
  soil_params = (θ_s, θ_r, Ks, α, n)

  # 模拟数据:5层土壤,深度0-50cm
  n_layers = 5
  depths = Float32.(collect(5:10:45))  # 5, 15, 25, 35, 45 cm
  Δz = 10.0f0  # cm
  Δt = 3600.0f0  # 1小时

  # 初始化 Soil 结构体
  soil = Soil{Float32}(n_layers=n_layers)
  soil.θ .= 0.25f0
  soil.θ_prev .= soil.θ

  # 观测值
  θ_obs = Float32.([0.26, 0.27, 0.26, 0.25, 0.24])  # 模拟观测

  # 初始化神经网络
  ps, st = Lux.setup(rng, nn_model)
  ps = ComponentArray(ps)
  dps = make_zero(ps)

  # 测试前向传播
  println("测试前向传播:")
  θ_pred = hybrid_forward(soil, depths, ps, st, soil_params, Δt, Δz)
  println("  预测含水量: ", round.(θ_pred, digits=4))
  println("  物理含水量: ", round.(soil.θ, digits=4))
  println("  节点传导度 K: ", round.(soil.K, digits=6))
  println("  界面传导度 K₊ₕ: ", round.(soil.K₊ₕ, digits=6))

  # 重新初始化 soil（因为前向传播修改了它）
  soil = Soil{Float32}(n_layers=n_layers)
  soil.θ .= 0.25f0
  soil.θ_prev .= soil.θ

  # 为梯度计算创建 soil 的副本
  dsoil = Soil{Float32}(n_layers=n_layers)

  # 测试梯度计算
  println("\n测试Enzyme梯度:")
  @time Enzyme.autodiff(
    set_runtime_activity(Reverse),  # 启用运行时活动分析
    loss_func,
    Active,  # 标量返回值用Active
    Duplicated(ps, dps),
    Duplicated(soil, dsoil),  # Soil 结构体也需要 Duplicated
    Const(depths), Const(θ_obs), Const(nn_model),
    Const(st), Const(soil_params),
    Const(Δt), Const(Δz)
  )

  println("\n梯度检查:")
  println("梯度范数: ", norm(dps))
  println("梯度样例: ", dps[1:min(5, length(dps))])
  println("是否有NaN: ", any(isnan, dps))
  println("是否有Inf: ", any(isinf, dps))

  println("\nSoil 梯度检查:")
  println("  dθ: ", round.(dsoil.θ, digits=6))
  println("  dK: ", round.(dsoil.K, digits=6))
end
