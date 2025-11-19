using Lux, Enzyme, ComponentArrays, Optimisers, Random


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
第二步:单时间步的显式Richards方程(最简化版本)
输入: θ_old (上一时刻含水量), 输出: θ_new (新时刻含水量)
=============================================================================#

function richards_explicit_step(θ_old, soil_params, Δt, Δz)
  θ_s, θ_r, Ks, α, n = soil_params
  n_layers = length(θ_old)
  θ_new = copy(θ_old)

  # 简单的向下流动(只考虑重力和基质势梯度)
  for i in 2:n_layers-1
    # 计算界面上的水力传导度(算术平均)
    K_upper = 0.5f0 * (hydraulic_conductivity(θ_old[i-1], θ_s, θ_r, Ks, n) +
                       hydraulic_conductivity(θ_old[i], θ_s, θ_r, Ks, n))
    K_lower = 0.5f0 * (hydraulic_conductivity(θ_old[i], θ_s, θ_r, Ks, n) +
                       hydraulic_conductivity(θ_old[i+1], θ_s, θ_r, Ks, n))

    # 水势梯度
    ψ_i = soil_water_potential(θ_old[i], θ_s, θ_r, α, n)
    ψ_upper = soil_water_potential(θ_old[i-1], θ_s, θ_r, α, n)
    ψ_lower = soil_water_potential(θ_old[i+1], θ_s, θ_r, α, n)

    # 达西通量(向下为正)
    q_upper = -K_upper * ((ψ_i - ψ_upper) / Δz + 1.0f0)
    q_lower = -K_lower * ((ψ_lower - ψ_i) / Δz + 1.0f0)

    # 更新含水量(通量散度)
    θ_new[i] = θ_old[i] + Δt / Δz * (q_upper - q_lower)
  end

  # 边界条件:顶层和底层保持不变(简化处理)
  θ_new[1] = θ_old[1]
  θ_new[end] = θ_old[end]

  return θ_new
end

#=============================================================================
第三步:定义混合模型 = Richards物理模型 + 神经网络修正
=============================================================================#

# 神经网络:输入是深度,输出是对Richards模型的修正项
nn_model = Chain(Dense(1 => 16, tanh), Dense(16 => 1))

# 完整的前向模型
function hybrid_forward(θ_init, depths, nn_params, st, soil_params, Δt, Δz)
  # 1. 物理模型预测
  θ_physics = richards_explicit_step(θ_init, soil_params, Δt, Δz)

  # 2. 神经网络修正(基于深度)
  corrections, _ = nn_model(depths', nn_params, st)  # depths: (n_layers,) -> (1, n_layers)
  corrections = vec(corrections)  # (n_layers,)

  # 3. 混合输出:physics + NN修正
  θ_hybrid = θ_physics .+ 0.01f0 .* corrections  # 小系数防止NN主导

  # 约束在合理范围
  θ_hybrid = clamp.(θ_hybrid, soil_params[2] + 0.01f0, soil_params[1] - 0.01f0)

  return θ_hybrid
end

#=============================================================================
第四步:损失函数(物理约束 + 数据拟合)
=============================================================================#

function loss_func(nn_params, θ_init, depths, θ_obs, nn_model, st, soil_params, Δt, Δz)
  # 前向预测
  θ_pred = hybrid_forward(θ_init, depths, nn_params, st, soil_params, Δt, Δz)

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

  # 初始含水量和观测值
  θ_init = fill(0.25f0, n_layers)
  θ_obs = Float32.([0.26, 0.27, 0.26, 0.25, 0.24])  # 模拟观测

  # 初始化神经网络
  ps, st = Lux.setup(rng, nn_model)
  ps = ComponentArray(ps)
  dps = make_zero(ps)

  # 测试前向传播
  println("测试前向传播:")
  θ_pred = hybrid_forward(θ_init, depths, ps, st, soil_params, Δt, Δz)
  @show θ_pred

  # 测试梯度计算
  println("\n测试Enzyme梯度:")
  @time Enzyme.autodiff(
    set_runtime_activity(Reverse),  # 启用运行时活动分析
    loss_func,
    Active,  # 标量返回值用Active
    Duplicated(ps, dps),
    Const(θ_init),
    Const(depths),
    Const(θ_obs),
    Const(nn_model),
    Const(st),
    Const(soil_params),
    Const(Δt),
    Const(Δz)
  )

  println("\n梯度检查:")
  println("梯度范数: ", norm(dps))
  println("梯度样例: ", dps[1:min(5, length(dps))])
  println("是否有NaN: ", any(isnan, dps))
  println("是否有Inf: ", any(isinf, dps))
end
