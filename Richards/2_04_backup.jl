using Lux, DifferentialEquations, DiffEqFlux, ComponentArrays, Random
using Parameters
using Statistics
using LinearAlgebra: norm
using Optimization, OptimizationOptimisers
using Zygote
using SciMLSensitivity  # 用于 sensealg
using CSV, DataFrames

include("Soil.jl")

#=============================================================================
NeuralODE 版本的物理信息神经网络（Physics-Informed Neural ODE）

核心思想：
- 使用 Richards 方程计算 dθ/dt（物理部分）
- 神经网络学习对 dθ/dt 的修正项（数据驱动部分）
- ODE 求解器自动处理时间积分
- 梯度通过 Adjoint 方法高效计算（内存复杂度 O(1)）

与 2_03.jl 的区别：
- 2_03: 离散时间步进（显式 Euler）
- 2_04: 连续时间 ODE（Tsit5 自适应求解器）
- 优势: 更高效、数值稳定、自适应步长

关于 AD 警告：
- Enzyme VJP 对 Lux + ODE 的组合支持有限，会自动回退到 ReverseDiff
- 我们显式指定 sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
- 这样可以避免警告并获得稳定的梯度计算
=============================================================================#


# Richards 方程时间导数计算（用于 ODE）
function richards_dθdt(θ, soil_params, Δz)
  θ_s, θ_r, Ks, α, n = soil_params
  n_layers = length(θ)

  # 限制含水量在合理范围内
  θ_clamp = clamp.(θ, θ_r + 0.001f0, θ_s - 0.001f0)

  # 计算每层的水力性质
  K = similar(θ)
  ψ = similar(θ)
  for i in 1:n_layers
    K[i] = hydraulic_conductivity(θ_clamp[i], θ_s, θ_r, Ks, n)
    ψ[i] = soil_water_potential(θ_clamp[i], θ_s, θ_r, α, n)
  end

  # 计算界面水力传导度（几何平均）
  K_interface = zeros(eltype(θ), n_layers + 1)
  K_interface[1] = K[1]
  for i in 2:n_layers
    K_interface[i] = sqrt(K[i-1] * K[i])  # 几何平均更稳定
  end
  K_interface[end] = K[end]

  # 计算界面通量
  Q = zeros(eltype(θ), n_layers + 1)
  for i in 2:n_layers
    Q[i] = -K_interface[i] * ((ψ[i] - ψ[i-1]) / Δz + 1.0f0)
  end

  # 边界通量
  Q[1] = 0.0f0      # 顶部无通量
  Q[end] = 0.0f0    # 底部无通量

  # 计算含水量变化率 dθ/dt
  dθdt = zeros(eltype(θ), n_layers)
  for i in 1:n_layers
    dθdt[i] = (Q[i] - Q[i+1]) / Δz
  end

  # 限制变化率（防止数值爆炸）
  dθdt = clamp.(dθdt, -0.001f0, 0.001f0)

  return dθdt
end

## 第二步: 定义混合模型 = Richards物理模型 + 神经网络修正
# 神经网络：输入是含水量，输出是对 dθ/dt 的修正项
nn_model = Chain(
  Dense(5 => 16, tanh),   # 输入是当前含水量 θ (5层)
  Dense(16 => 16, tanh),
  Dense(16 => 5)          # 输出是修正项 (5层)
)

# 混合 ODE 函数：physics + NN 修正
# 这是 NeuralODE 所需的 dudt 函数
function create_hybrid_ode(nn_model, nn_state, depths, soil_params, Δz)
  function hybrid_ode!(dθ, θ, p, t)
    θ_s, θ_r = soil_params[1], soil_params[2]

    # 限制含水量在合理范围
    θ_safe = clamp.(θ, θ_r + 0.001f0, θ_s - 0.001f0)

    # 1. 物理模型：简化的Richards方程（非常小的权重）
    dθ_physics = richards_dθdt(θ_safe, soil_params, Δz)

    # 2. 神经网络修正：基于当前状态 θ
    θ_input = reshape(θ_safe, :, 1)  # (n_layers,) -> (n_layers, 1)

    # 使用 Lux 模型（使用捕获的状态）
    corrections, _ = nn_model(θ_input, p, nn_state)  # 返回 (n_layers, 1)
    corrections = vec(corrections)  # (n_layers,)

    # 3. 混合：主要依赖NN，物理项权重很小
    dθ_hybrid = 0.001f0 .* dθ_physics .+ 0.0001f0 .* corrections

    # 限制变化率
    dθ_hybrid = clamp.(dθ_hybrid, -0.0001f0, 0.0001f0)

    dθ .= dθ_hybrid # 更新 dθ（原地修改）
    return nothing
  end
  return hybrid_ode!
end

## 第三步: 定义 NeuralODE 层
# 创建 NeuralODE 包装器
struct HybridNeuralODE{M,T,P,S}
  nn_model::M
  nn_state::S          # 添加状态
  tspan::T
  depths::Vector{Float32}
  soil_params::P
  Δz::Float32
  alg
  kwargs
end


function HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params, Δz;
  alg=Tsit5(), kwargs...)
  return HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params, Δz, alg, kwargs)
end

# 前向传播：求解 ODE
function (node::HybridNeuralODE)(θ₀, p, st)
  # 创建混合 ODE 函数（使用存储的 nn_state）
  ode_func! = create_hybrid_ode(node.nn_model, node.nn_state, node.depths, node.soil_params, node.Δz)

  prob = ODEProblem(ode_func!, θ₀, node.tspan, p)
  # 确保输出点数与观测匹配：从t_start开始，每小时一个点，共n_train个
  n_save = Int((node.tspan[2] - node.tspan[1]) / 3600.0f0)
  saveat = range(node.tspan[1], node.tspan[2], length=n_save+1)[1:end-1]  # 排除最后一个点
  sol = solve(prob, node.alg; node.kwargs..., saveat=saveat)
  return sol, st
end


## 第四步: 损失函数（数据拟合 + 物理约束）
function loss_neuralode(p, θ₀, θ_obs_seq, node, soil_params)
  # θ_obs_seq: (n_layers, n_times) 观测序列
  θ_s, θ_r = soil_params[1], soil_params[2]

  # 前向求解 ODE
  sol, _ = node(θ₀, p, NamedTuple())
  θ_pred_seq = Array(sol)  # (n_layers, n_times)

  # 数据拟合损失
  loss_data = mean((θ_pred_seq .- θ_obs_seq) .^ 2)

  # 物理约束 1：含水量不能超出饱和度
  loss_physics1 = mean(max.(0.0f0, θ_pred_seq .- θ_s) .^ 2)

  # 物理约束 2：含水量不能低于残余含水量
  loss_physics2 = mean(max.(0.0f0, θ_r .- θ_pred_seq) .^ 2)

  total_loss = loss_data + 0.1f0 * (loss_physics1 + loss_physics2)
  return total_loss
end


## 第五步: 加载真实观测数据
function load_soil_data(file_path)
  df = CSV.read(file_path, DataFrame)

  # 提取土壤水分观测 (5层: 5, 10, 20, 50, 100 cm)
  θ_obs = Matrix{Float32}(df[:, [:SOIL_MOISTURE_5, :SOIL_MOISTURE_10, :SOIL_MOISTURE_20,
                                   :SOIL_MOISTURE_50, :SOIL_MOISTURE_100]])

  # 提取降水 [mm] -> [cm]
  P = Float32.(df.P_CALC ./ 10.0)

  return θ_obs', P  # 转置为 (n_layers, n_times)
end

## 第六步: 测试 NeuralODE（使用真实数据）
begin
  # 加载真实数据
  data_file = joinpath(@__DIR__, "..", "data", "SM_AR_Batesville_8_WNW_2024.csv")
  θ_obs_full, P_full = load_soil_data(data_file)

  println("\n[加载真实数据]")
  println("数据维度: $(size(θ_obs_full)), 时间步数: $(size(θ_obs_full, 2))")
  println("土壤水分范围: [$(round(minimum(θ_obs_full), digits=3)), $(round(maximum(θ_obs_full), digits=3))]")
  println("降水范围: [$(round(minimum(P_full), digits=3)), $(round(maximum(P_full), digits=3))] cm")

  # 选择训练时段：前10天 (240小时)
  n_train = 240
  θ_obs_train = θ_obs_full[:, 1:n_train]
  θ₀ = θ_obs_train[:, 1]  # 使用真实初始值

  println("训练数据: $(size(θ_obs_train)), 初始含水量: $(round.(θ₀, digits=3))")

  # 土壤参数（根据数据范围估计）
  θ_s = 0.45f0    # 饱和含水量
  θ_r = 0.05f0    # 残余含水量
  Ks = 0.01f0     # 饱和导水率 cm/s
  α = 0.02f0      # cm^-1
  n = 1.5f0
  soil_params = (θ_s, θ_r, Ks, α, n)

  # 模拟设置：5层土壤
  n_layers = 5
  depths = Float32.([5, 10, 20, 50, 100])  # cm
  Δz = 10.0f0  # cm (平均层厚)

  # 时间设置：模拟10天，每小时一个数据点
  t_start = 0.0f0
  t_end = Float32(n_train * 3600)  # 转换为秒
  tspan = (t_start, t_end)

  # 初始化神经网络
  rng = Xoshiro(123)
  ps, st = Lux.setup(rng, nn_model)
  ps = ComponentArray(ps)

  println("\n[测试 NeuralODE 前向传播]")
  # 显式指定使用 InterpolatingAdjoint 和 ReverseDiffVJP
  hybrid_node = HybridNeuralODE(
    nn_model, st, tspan, depths, soil_params, Δz;
    alg=Tsit5(), reltol=1e-4, abstol=1e-6,
    sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
    verbose=false  # 关闭警告信息
  )

  # 测试前向传播
  sol, _ = hybrid_node(θ₀, ps, st)
  println("前向传播: n_times=$(length(sol.t)), θ_init=$(round.(sol.u[1], digits=3)), θ_final=$(round.(sol.u[end], digits=3))")

  # 使用真实观测数据
  θ_obs_seq = θ_obs_train
  println("观测数据: size=$(size(θ_obs_seq)), range=[$(round(minimum(θ_obs_seq), digits=3)), $(round(maximum(θ_obs_seq), digits=3))]")

  # 计算损失和评估指标
  loss_val = loss_neuralode(ps, θ₀, θ_obs_seq, hybrid_node, soil_params)

  # 提取模拟结果进行评估
  θ_pred_seq = Array(sol)  # (n_layers, n_times)

  # 计算 RMSE 和 R²
  function calc_metrics(pred, obs)
    rmse = sqrt(mean((pred .- obs) .^ 2))
    ss_res = sum((obs .- pred) .^ 2)
    ss_tot = sum((obs .- mean(obs)) .^ 2)
    r2 = 1 - ss_res / ss_tot
    return rmse, r2
  end

  rmse_all, r2_all = calc_metrics(θ_pred_seq, θ_obs_seq)

  println("损失值: $(round(loss_val, digits=6))")
  println("整体评估: RMSE=$(round(rmse_all, digits=4)), R²=$(round(r2_all, digits=4))")

  # 各层评估
  println("\n各层模拟效果:")
  for i in 1:n_layers
    rmse_i, r2_i = calc_metrics(θ_pred_seq[i, :], θ_obs_seq[i, :])
    println("  $(Int(depths[i]))cm: RMSE=$(round(rmse_i, digits=4)), R²=$(round(r2_i, digits=4))")
  end

  ## 测试梯度计算（使用 Zygote）
  println("\n[测试 NeuralODE 梯度计算]")

  # 定义损失函数的包装器（固定部分参数）
  loss_wrapper(p) = loss_neuralode(p, θ₀, θ_obs_seq, hybrid_node, soil_params)

  # 注意：由于我们显式指定了 sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))
  # 梯度将通过 adjoint 方法计算，这比朴素的反向传播更高效
  @time grads = Zygote.gradient(loss_wrapper, ps)
  grad_norm = norm(grads[1])
  has_nan = any(isnan, grads[1])
  has_inf = any(isinf, grads[1])
  grad_sample = round.(grads[1][1:min(5, length(grads[1]))], digits=8)

  println("梯度: norm=$(round(grad_norm, digits=6)), sample=$grad_sample, NaN=$has_nan, Inf=$has_inf")
  if grad_norm > 0.0 && !has_nan && !has_inf
    println("✓ 梯度计算成功")
  else
    println("⚠ 梯度异常")
  end
end
