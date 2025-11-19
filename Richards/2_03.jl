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

# 单时刻前向模型：使用 Soil 结构体
function hybrid_forward(soil::Soil, depths, nn_params, st, soil_params, Δt, Δz)
  # 1. 物理模型预测（原地修改）
  richards_explicit_step!(soil, soil_params, Δt, Δz)

  # 2. 神经网络修正(基于深度)
  corrections, _ = nn_model(depths', nn_params, st)  # depths: (n_layers,) -> (1, n_layers)
  corrections = vec(corrections)  # (n_layers,)

  # 3. 混合输出:physics + NN修正
  θ_hybrid = soil.θ .+ 0.01f0 .* corrections  # 小系数防止NN主导

  # 约束在合理范围
  θ_s, θ_r = soil_params[1], soil_params[2]
  θ_hybrid = clamp.(θ_hybrid, θ_r + 0.01f0, θ_s - 0.01f0)

  return θ_hybrid
end

# 多时刻前向模型：循环多个时间步
function hybrid_forward_multistep(soil::Soil, depths, nn_params, st, soil_params, Δt, Δz, n_times::Int)
  n_layers = soil.n_layers
  θ_s, θ_r = soil_params[1], soil_params[2]

  # 初始化输出数组 (n_layers, n_times)
  θ_predictions = zeros(eltype(soil.θ), n_layers, n_times)

  # 循环模拟每个时间步
  for t in 1:n_times
    # 1. 物理模型预测（原地修改 soil 状态）
    richards_explicit_step!(soil, soil_params, Δt, Δz)

    # 2. 神经网络修正(基于深度)
    corrections, _ = nn_model(depths', nn_params, st)
    corrections = vec(corrections)

    # 3. 混合输出:physics + NN修正
    θ_hybrid = soil.θ .+ 0.01f0 .* corrections

    # 约束在合理范围
    θ_hybrid = clamp.(θ_hybrid, θ_r + 0.01f0, θ_s - 0.01f0)

    # 保存当前时刻的预测
    θ_predictions[:, t] .= θ_hybrid
  end

  return θ_predictions
end

#=============================================================================
第四步:损失函数(物理约束 + 数据拟合)
=============================================================================#
# 单时刻损失函数
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

# 多时刻损失函数
function loss_func_multistep(nn_params, soil, depths, θ_obs_seq, nn_model, st, soil_params, Δt, Δz)
  # θ_obs_seq: (n_layers, n_times) 观测序列
  n_times = size(θ_obs_seq, 2)

  # 前向预测多个时间步
  θ_pred_seq = hybrid_forward_multistep(soil, depths, nn_params, st, soil_params, Δt, Δz, n_times)

  # 数据拟合损失：对所有时刻和所有层求平均
  loss_data = mean((θ_pred_seq .- θ_obs_seq) .^ 2)

  # 物理约束:含水量不能超出饱和度
  θ_s = soil_params[1]
  loss_physics = mean(max.(0.0f0, θ_pred_seq .- θ_s) .^ 2)

  return loss_data + 0.1f0 * loss_physics
end

#=============================================================================
第五步:测试梯度计算（多时刻版本）
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
  n_times = 10  # 模拟10个时间步

  # 初始化神经网络
  ps, st = Lux.setup(rng, nn_model)
  ps = ComponentArray(ps)
  dps = make_zero(ps)

  #===========================================================================
  测试1: 单时刻版本（保留原有测试）
  ===========================================================================#
  println("="^70)
  println("测试1: 单时刻前向传播和梯度")
  println("="^70)

  # 初始化 Soil 结构体
  soil_single = Soil{Float32}(n_layers=n_layers)
  soil_single.θ .= 0.25f0
  soil_single.θ_prev .= soil_single.θ

  # 单时刻观测值
  θ_obs_single = Float32.([0.26, 0.27, 0.26, 0.25, 0.24])

  # 测试前向传播
  println("\n单时刻前向传播:")
  θ_pred_single = hybrid_forward(soil_single, depths, ps, st, soil_params, Δt, Δz)
  println("  预测含水量: ", round.(θ_pred_single, digits=4))
  println("  物理含水量: ", round.(soil_single.θ, digits=4))

  # 重新初始化用于梯度测试
  soil_single = Soil{Float32}(n_layers=n_layers)
  soil_single.θ .= 0.25f0
  soil_single.θ_prev .= soil_single.θ
  dsoil_single = Soil{Float32}(n_layers=n_layers)

  # 测试单时刻梯度
  println("\n单时刻梯度计算:")
  dps .= 0.0f0  # 重置梯度
  @time Enzyme.autodiff(
    set_runtime_activity(Reverse),
    loss_func,
    Active,
    Duplicated(ps, dps),
    Duplicated(soil_single, dsoil_single),
    Const(depths), Const(θ_obs_single), Const(nn_model),
    Const(st), Const(soil_params),
    Const(Δt), Const(Δz)
  )

  println("  梯度范数: ", round(norm(dps), digits=6))
  println("  是否有NaN: ", any(isnan, dps))
  println("  是否有Inf: ", any(isinf, dps))

  #===========================================================================
  测试2: 多时刻版本（新增）
  ===========================================================================#
  println("\n" * "="^70)
  println("测试2: 多时刻前向传播和梯度")
  println("="^70)

  # 初始化 Soil 结构体
  soil_multi = Soil{Float32}(n_layers=n_layers)
  soil_multi.θ .= 0.25f0
  soil_multi.θ_prev .= soil_multi.θ

  # 生成模拟的多时刻观测数据 (n_layers, n_times)
  # 模拟含水量随时间缓慢增加的情况
  θ_obs_multi = zeros(Float32, n_layers, n_times)
  for t in 1:n_times
    # 不同深度不同变化趋势
    for i in 1:n_layers
      base = 0.25f0
      trend = 0.01f0 * (t - 1) / n_times  # 缓慢增加
      depth_effect = 0.02f0 * (1.0f0 - (i - 1) / (n_layers - 1))  # 浅层含水量较高
      noise = 0.005f0 * randn(rng, Float32)
      θ_obs_multi[i, t] = base + trend + depth_effect + noise
    end
  end

  # 约束观测值在合理范围内
  θ_obs_multi .= clamp.(θ_obs_multi, θ_r + 0.01f0, θ_s - 0.01f0)

  println("\n多时刻观测数据形状: ", size(θ_obs_multi))
  println("观测值范围: [", round(minimum(θ_obs_multi), digits=4), ", ",
    round(maximum(θ_obs_multi), digits=4), "]")

  # 测试多时刻前向传播
  println("\n多时刻前向传播:")
  θ_pred_multi = hybrid_forward_multistep(soil_multi, depths, ps, st, soil_params, Δt, Δz, n_times)
  println("  预测数据形状: ", size(θ_pred_multi))
  println("  第1时刻预测: ", round.(θ_pred_multi[:, 1], digits=4))
  println("  第$(n_times)时刻预测: ", round.(θ_pred_multi[:, end], digits=4))
  println("  预测值范围: [", round(minimum(θ_pred_multi), digits=4), ", ",
    round(maximum(θ_pred_multi), digits=4), "]")

  # 重新初始化用于梯度测试
  soil_multi = Soil{Float32}(n_layers=n_layers)
  soil_multi.θ .= 0.25f0
  soil_multi.θ_prev .= soil_multi.θ
  dsoil_multi = Soil{Float32}(n_layers=n_layers)

  # 测试多时刻梯度
  println("\n多时刻梯度计算:")
  dps .= 0.0f0  # 重置梯度
  @time Enzyme.autodiff(
    set_runtime_activity(Reverse),
    loss_func_multistep,
    Active,
    Duplicated(ps, dps),
    Duplicated(soil_multi, dsoil_multi),
    Const(depths), Const(θ_obs_multi), Const(nn_model),
    Const(st), Const(soil_params),
    Const(Δt), Const(Δz)
  )

  println("  梯度范数: ", round(norm(dps), digits=6))
  println("  梯度样例: ", round.(dps[1:min(5, length(dps))], digits=8))
  println("  是否有NaN: ", any(isnan, dps))
  println("  是否有Inf: ", any(isinf, dps))

  println("\n  Soil 梯度检查:")
  println("    dθ: ", round.(dsoil_multi.θ, digits=6))
  println("    dK: ", round.(dsoil_multi.K, digits=6))

  println("\n" * "="^70)
  println("测试完成！")
  println("="^70)
end
