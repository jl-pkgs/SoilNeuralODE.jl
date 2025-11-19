using Lux, DifferentialEquations, DiffEqFlux, ComponentArrays, Random
using Parameters
using Statistics
using LinearAlgebra: norm
using Optimization, OptimizationOptimisers
using Zygote
using SciMLSensitivity  # 用于 sensealg

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

  # 计算每层的水力性质
  K = similar(θ)
  ψ = similar(θ)
  for i in 1:n_layers
    K[i] = hydraulic_conductivity(θ[i], θ_s, θ_r, Ks, n)
    ψ[i] = soil_water_potential(θ[i], θ_s, θ_r, α, n)
  end

  # 计算界面水力传导度
  K_interface = zeros(eltype(θ), n_layers + 1)
  K_interface[1] = K[1]
  for i in 2:n_layers
    K_interface[i] = 0.5f0 * (K[i-1] + K[i])
  end
  K_interface[end] = K[end]

  # 计算界面通量
  Q = zeros(eltype(θ), n_layers + 1)
  for i in 2:n_layers
    Q[i] = -K_interface[i] * ((ψ[i] - ψ[i-1]) / Δz + 1.0f0)
  end

  # 边界通量（无通量边界）
  Q[1] = 0.0f0
  Q[end] = 0.0f0

  # 计算含水量变化率 dθ/dt
  dθdt = zeros(eltype(θ), n_layers)
  for i in 1:n_layers
    dθdt[i] = (Q[i] - Q[i+1]) / Δz
  end

  # 边界条件：顶层和底层不变
  dθdt[1] = 0.0f0
  dθdt[end] = 0.0f0

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

    # 1. 物理模型：Richards 方程
    dθ_physics = richards_dθdt(θ, soil_params, Δz)

    # 2. 神经网络修正：基于当前状态 θ
    θ_input = reshape(θ, :, 1)  # (n_layers,) -> (n_layers, 1)

    # 使用 Lux 模型（使用捕获的状态）
    corrections, _ = nn_model(θ_input, p, nn_state)  # 返回 (n_layers, 1)
    corrections = vec(corrections)  # (n_layers,)

    # 3. 混合：物理 + 小系数的 NN 修正
    dθ_hybrid = dθ_physics .+ 0.01f0 .* corrections
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
  sol = solve(prob, node.alg; node.kwargs..., saveat=node.tspan[1]:3600.0f0:node.tspan[2])
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


## 第五步: 测试 NeuralODE（前向传播 + 梯度）
begin
  # 土壤参数（典型壤土）
  θ_s = 0.45f0    # 饱和含水量
  θ_r = 0.05f0    # 残余含水量
  Ks = 0.01f0     # 饱和导水率 cm/s
  α = 0.02f0      # cm^-1
  n = 1.5f0
  soil_params = (θ_s, θ_r, Ks, α, n)

  # 模拟设置：5层土壤
  n_layers = 5
  depths = Float32.(collect(5:10:45))  # 5, 15, 25, 35, 45 cm
  Δz = 10.0f0  # cm

  # 时间设置：模拟10小时
  t_start = 0.0f0
  t_end = 36000.0f0  # 10小时 = 36000秒
  tspan = (t_start, t_end)

  θ₀ = Float32.([0.25, 0.25, 0.25, 0.25, 0.25]) # 初始条件

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
  ) # 创建 HybridNeuralODE（传入 nn_state）

  # 测试前向传播
  sol, _ = hybrid_node(θ₀, ps, st)
  println("前向传播: n_times=$(length(sol.t)), θ_init=$(round.(sol.u[1], digits=4)), θ_final=$(round.(sol.u[end], digits=4))")

  # 生成模拟观测数据
  n_times = length(sol.t)
  θ_obs_seq = Array(sol) .+ 0.01f0 .* randn(rng, Float32, n_layers, n_times)
  θ_obs_seq .= clamp.(θ_obs_seq, θ_r + 0.01f0, θ_s - 0.01f0)
  println("观测数据: size=$(size(θ_obs_seq)), range=[$(round(minimum(θ_obs_seq), digits=4)), $(round(maximum(θ_obs_seq), digits=4))]")

  # 计算损失
  loss_val = loss_neuralode(ps, θ₀, θ_obs_seq, hybrid_node, soil_params)
  println("损失值: $(round(loss_val, digits=6))")

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
