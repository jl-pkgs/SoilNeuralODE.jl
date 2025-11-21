export richards_dθdt, HybridNeuralODE
export loss_mse, train, predict, evaluate

using Optimization, OptimizationOptimisers


# 计算实际入渗通量（物理基础 + 神经网络修正）
# 单位：P_flux [cm/h], θ [-], K_surface [cm/h] -> Q_infiltration [cm/h]
function infiltration(P_flux, θ_surface, K_surface, nn_model, p, nn_state)
  # 物理基础：入渗不超过饱和导水率
  Q_base = min(P_flux, K_surface)

  # 神经网络学习入渗修正项
  # 输入：[P, θ_surface, K_surface]
  nn_input = reshape([P_flux, θ_surface, K_surface], 3, 1)
  Q_correction_out, _ = nn_model(nn_input, p, nn_state)
  Q_correction = Q_correction_out[1]

  # 实际入渗 = 物理基础 + 修正项
  Q_infiltration = Q_base + 0.1f0 * Q_correction
  return Q_infiltration
end


# 辅助函数：平滑限制器 (替代 output clamp)
# 保持导数平滑，防止梯度消失
@inline function soft_limit(x::T, limit::T) where T
  return limit * tanh(x / limit)
end

# Richards方程：改为 In-Place 写法
# du (dθdt) 由外部传入并在内部修改
function richards_dθdt!(dθ, θ, depths, soil_params_profile, Q_infiltration)
  n = length(θ)

  # 在 Enzyme 中，建议尽量减少内部临时数组分配
  # 这里为了简化演示保留了 K, ψ, Q 的分配，
  # 极致优化可以把这些也作为缓存传入 (Pre-allocation)
  K = similar(θ)
  ψ = similar(θ)
  Q = similar(θ, n + 1)

  # 1. 计算水力参数
  for i in 1:n
    θ_s, θ_r, Ks, α, n_vg = soil_params_profile[i]

    # Input Clamp: 状态量的物理约束可以用硬 clamp
    # 确保类型一致 (Float32)，避免 Enzyme 类型推断困难
    θ_safe = clamp(θ[i], θ_r + 1f-3, θ_s - 1f-3)

    K[i] = hydraulic_conductivity(θ_safe, θ_s, θ_r, Ks, n_vg)
    ψ[i] = soil_water_potential(θ_safe, θ_s, θ_r, α, n_vg)
  end

  # 2. 计算通量 Q
  Q[1] = Q_infiltration

  for i in 2:n
    Δz = depths[i] - depths[i-1]
    # 加上 1f-10 防止除以0，增加数值稳定性
    denom = K[i-1] + K[i] + 1f-10
    K_interface = 2.0f0 * K[i-1] * K[i] / denom
    ψ_gradient = (ψ[i] - ψ[i-1]) / Δz
    Q[i] = -K_interface * (ψ_gradient + 1.0f0)
  end
  Q[n+1] = -K[n] # 自由排水

  # 3. 计算 dθ (In-Place 修改)
  for i in 1:n
    Δz_cell = if i == 1
      depths[2] - depths[1]
    elseif i == n
      depths[n] - depths[n-1]
    else
      (depths[i+1] - depths[i-1]) / 2.0f0
    end

    val = -(Q[i+1] - Q[i]) / Δz_cell

    # --- 关键修改 ---
    # 使用 Soft Limit 替代 hard clamp
    # 这样即使 val 很大，梯度也能传回去告诉网络减小输入
    dθ[i] = soft_limit(val, 0.001f0)
  end
  return nothing
end

# 混合 ODE 创建函数
function create_hybrid_ode(nn_model, nn_state, depths, soil_params_profile, P_interp)
  # 标准 SciML In-Place 签名: f(du, u, p, t)
  function ode!(dθ, θ, p, t)
    t_value = float(t)
    P_flux = P_interp(t_value)

    # 计算 θ_safe (用于输入神经网络和 K_surface)
    # 这里只取第一层，使用 Base.clamp 即可，Enzyme 支持
    # 注意：不要 broadcat 到整个数组，只算需要的，减少计算量
    sp1 = soil_params_profile[1]
    θ_s, θ_r, Ks, α, n_vg = sp1
    θ_safe_surf = clamp(θ[1], θ_r + 1f-3, θ_s - 1f-3)

    K_surface = hydraulic_conductivity(θ_safe_surf, θ_s, θ_r, Ks, n_vg)

    # 神经网络推理
    Q_infiltration = infiltration(P_flux, θ_safe_surf, K_surface, nn_model, p, nn_state)

    # 调用 In-Place 的 Richards 方程
    richards_dθdt!(dθ, θ, depths, soil_params_profile, Q_infiltration)

    return nothing
  end
  return ode!
end

# NeuralODE 包装器
struct HybridNeuralODE{M,T,P,S,F}
  nn_model::M
  nn_state::S
  tspan::T
  depths::Vector{Float32}
  # Vector of tuples: [(θ_s, θ_r, Ks, α, n) for each layer]
  soil_params_profile::P  
  P_interp::F  # 降水插值函数
  alg
  kwargs
end


function HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params_profile, P_interp;
                         alg=Tsit5(), kwargs...)
  return HybridNeuralODE(nn_model, nn_state, tspan, depths, soil_params_profile,
                        P_interp, alg, kwargs)
end


function (node::HybridNeuralODE)(θ₀, p, st)
  ode_func! = create_hybrid_ode(node.nn_model, node.nn_state, node.depths,
                                node.soil_params_profile, node.P_interp)
  prob = ODEProblem(ode_func!, θ₀, node.tspan, p)

  # tspan现在是小时，每小时保存一次
  n_save = Int(node.tspan[2] - node.tspan[1])
  saveat = range(node.tspan[1], node.tspan[2], length=n_save+1)[1:end-1]
  sol = solve(prob, node.alg; node.kwargs..., saveat=saveat)
  return sol, st
end
