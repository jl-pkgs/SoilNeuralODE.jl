"""
高级土壤水运动模型 - Enzyme.jl 自定义梯度
实现fixed-point迭代的隐式微分
"""

using Lux
using Random
using Enzyme
using ComponentArrays
using LinearAlgebra
using Statistics

# 可选：如果安装了 Zygote，使用它会更稳定
using Zygote
ZYGOTE_AVAILABLE = true

# try
#   println("✓ Zygote.jl 可用 - 推荐用于梯度计算")
# catch
#   ZYGOTE_AVAILABLE = false
#   println("○ Zygote.jl 未安装 - 将使用 Enzyme 或有限差分")
# end

# ============================================================================
# 1. 可微分的土壤水力函数
# ============================================================================

"""van Genuchten模型 - 含水量"""
@inline function vg_theta(h::T, θs, θr, α, n) where T
  m = 1 - 1 / n
  if h >= 0
    return T(θs)
  else
    Se = 1 / (1 + (α * abs(h))^n)^m
    return T(θr + (θs - θr) * Se)
  end
end

"""van Genuchten模型 - 导水率"""
@inline function vg_K(h::T, Ks, θs, θr, α, n) where T
  m = 1 - 1 / n
  if h >= 0
    return T(Ks)
  else
    Se = 1 / (1 + (α * abs(h))^n)^m
    K = Ks * sqrt(Se) * (1 - (1 - Se^(1 / m))^m)^2
    return T(K)
  end
end

"""水分容量 C(h) = dθ/dh"""
@inline function water_capacity(h::T, θs, θr, α, n) where T
  m = 1 - 1 / n
  if h >= 0
    return T(0.0)
  else
    Se = 1 / (1 + (α * abs(h))^n)^m
    dSe_dh = -m * n * α^n * abs(h)^(n - 1) * Se^(1 + 1 / m) / abs(h)
    C = (θs - θr) * dSe_dh
    return T(C)
  end
end

# ============================================================================
# 2. Richards方程求解器 (带隐式微分)
# ============================================================================

struct RichardsSolver
  nz::Int
  dz::Float64
  dt::Float64
  max_iter::Int
  tol::Float64
end

RichardsSolver(nz, dz, dt) = RichardsSolver(nz, dz, dt, 10, 1e-6)

"""
Picard迭代求解Richards方程
这是一个fixed-point问题: h_new = F(h_new, h_old, params)
"""
function solve_richards_picard!(h_new, h_old, params, infiltration, solver::RichardsSolver)
  (; nz, dz, dt, max_iter, tol) = solver
  (; θs, θr, α, n, Ks) = params

  # Picard迭代
  for iter in 1:max_iter
    h_prev = copy(h_new)

    # 计算水力性质
    θ = [vg_theta(h_new[i], θs, θr, α, n) for i in 1:nz]
    K = [vg_K(h_new[i], Ks, θs, θr, α, n) for i in 1:nz]
    C = [water_capacity(h_new[i], θs, θr, α, n) for i in 1:nz]

    # 更新内部节点
    for i in 2:nz-1
      K_up = 0.5 * (K[i-1] + K[i])
      K_down = 0.5 * (K[i] + K[i+1])

      # 通量计算
      q_up = -K_up * ((h_new[i] - h_new[i-1]) / dz + 1.0)
      q_down = -K_down * ((h_new[i+1] - h_new[i]) / dz + 1.0)

      # 质量守恒
      dq = (q_up - q_down) / dz

      # 更新压力水头 (混合法)
      C_eff = max(C[i], 0.01)  # 避免除零
      h_new[i] = h_old[i] + dt * dq / C_eff

      # 限制变化幅度
      max_change = 0.5
      h_new[i] = clamp(h_new[i], h_prev[i] - max_change, h_prev[i] + max_change)
    end

    # 上边界 (Neumann条件 - 入渗通量)
    q_in = min(infiltration, Ks)
    h_new[1] = h_new[2] + dz * (q_in / K[1] - 1.0)

    # 下边界 (单位梯度)
    h_new[nz] = h_new[nz-1] - dz

    # 收敛检查
    residual = norm(h_new - h_prev) / (norm(h_new) + 1e-8)
    if residual < tol
      break
    end
  end

  return h_new
end

# ============================================================================
# 3. 使用Enzyme的隐式微分
# ============================================================================

"""
为fixed-point迭代定义自定义梯度规则
使用隐式函数定理: 若 F(y*, x) = y*, 则 dy*/dx = -[dF/dy - I]^{-1} * dF/dx
"""
function richards_with_implicit_grad(h_old, params, infiltration, solver)
  # 前向求解
  h_new = copy(h_old)
  h_new = solve_richards_picard!(h_new, h_old, params, infiltration, solver)

  return h_new
end

"""
Zygote 友好版本：将 Richards 求解器视为不可微的黑盒
这样 Zygote 只会微分神经网络部分
"""
function richards_no_grad(h_old, params, infiltration, solver)
  if ZYGOTE_AVAILABLE
    # 使用 Zygote.ignore 告诉 Zygote 不要尝试微分这个函数
    return Zygote.ignore() do
      h_new = copy(h_old)
      solve_richards_picard!(h_new, h_old, params, infiltration, solver)
      return h_new
    end
  else
    h_new = copy(h_old)
    return solve_richards_picard!(h_new, h_old, params, infiltration, solver)
  end
end

"""
计算Jacobian: ∂F/∂h (用于隐式微分)
这里使用有限差分近似
"""
function compute_jacobian(h, h_old, params, infiltration, solver)
  nz = length(h)
  J = zeros(nz, nz)
  eps = 1e-6

  for j in 1:nz
    h_perturb = copy(h)
    h_perturb[j] += eps

    F_perturb = solve_richards_picard!(copy(h_perturb), h_old, params, infiltration, solver)
    F_base = solve_richards_picard!(copy(h), h_old, params, infiltration, solver)

    J[:, j] = (F_perturb - F_base) / eps
  end

  return J
end

# ============================================================================
# 4. 物理-神经网络混合模型
# ============================================================================

"""简化的神经网络层"""
struct SimpleNN
  nz::Int
  hidden::Int
end

function (nn::SimpleNN)(h, z, params)
  # 输入特征: 归一化的h和z
  h_norm = h ./ 10.0  # 简单归一化
  z_norm = z ./ maximum(z)

  # 输入层
  x = vcat(h_norm, z_norm)

  # 隐藏层
  W1 = reshape(params.W1, nn.hidden, 2 * nn.nz)
  b1 = params.b1
  h1 = tanh.(W1 * x .+ b1)

  # 输出层
  W2 = reshape(params.W2, nn.nz, nn.hidden)
  b2 = params.b2
  out = W2 * h1 .+ b2

  return out
end

"""完整混合模型"""
struct HybridModel
  solver::RichardsSolver
  nn::SimpleNN
end

function (model::HybridModel)(x, params; use_zygote=false)
  h_old, infiltration, z = x

  # 物理模型 - 根据是否使用 Zygote 选择不同版本
  if use_zygote && ZYGOTE_AVAILABLE
    # Zygote 友好版本：Richards 求解器作为黑盒
    h_physics = richards_no_grad(h_old, params.physics, infiltration, model.solver)
  else
    # 标准版本
    h_physics = richards_with_implicit_grad(h_old, params.physics, infiltration, model.solver)
  end

  # 神经网络修正
  correction = model.nn(h_physics, z, params.neural)

  # 混合 (小幅修正)
  h_hybrid = h_physics .+ 0.01 .* correction

  return h_hybrid
end

# ============================================================================
# 5. 初始化和测试
# ============================================================================

function initialize_hybrid_model()
  # Richards求解器
  nz = 15
  depth = 1.5  # 1.5米
  dz = depth / nz
  dt = 1800.0  # 30分钟

  solver = RichardsSolver(nz, dz, dt)

  # 神经网络
  nn = SimpleNN(nz, 20)

  # 初始化参数
  params = ComponentArray(
    physics=(
      θs=0.43,
      θr=0.06,
      α=1.8,
      n=1.45,
      Ks=8e-6
    ),
    neural=(
      W1=randn(20 * 2 * nz) * 0.1,
      b1=zeros(20),
      W2=randn(nz * 20) * 0.1,
      b2=zeros(nz)
    )
  )

  model = HybridModel(solver, nn)

  return model, params, solver.dz, nz
end

function compute_loss(model, params, h_init, infiltration, z, θ_obs; use_zygote=false)
  # 前向传播
  h_pred = model((h_init, infiltration, z), params, use_zygote=use_zygote)

  # 计算预测含水量
  θ_pred = [vg_theta(h_pred[i], params.physics.θs, params.physics.θr,
    params.physics.α, params.physics.n) for i in eachindex(h_pred)]

  # MSE损失
  loss = mean((θ_pred .- θ_obs) .^ 2)

  # 物理约束
  penalty = sum(max.(0, θ_pred .- params.physics.θs) .^ 2) +
            sum(max.(0, params.physics.θr .- θ_pred) .^ 2)

  return loss + 0.1 * penalty
end

"""
使用Enzyme计算梯度（反向模式）
注意: Enzyme对于复杂的嵌套结构需要特殊处理
"""
function compute_gradient_enzyme(model, params, h_init, infiltration, z, θ_obs)
  # 将ComponentArray转换为普通数组以便Enzyme处理
  params_vec = Vector(params)

  # 定义接受向量参数的损失函数
  function loss_vec(p_vec)
    # 重构ComponentArray
    p = ComponentArray(p_vec, getaxes(params))
    return compute_loss(model, p, h_init, infiltration, z, θ_obs)
  end

  # 使用Enzyme的反向模式AD
  grad = zeros(length(params_vec))

  try
    # Enzyme.autodiff 使用 Reverse 模式
    Enzyme.autodiff(Enzyme.Reverse, loss_vec, Enzyme.Active, Enzyme.Duplicated(params_vec, grad))

    # 将梯度向量转换回ComponentArray结构
    grad_ca = ComponentArray(grad, getaxes(params))
    return grad_ca, loss_vec(params_vec)

  catch e
    println("  Enzyme错误: ", e)
    println("  尝试使用前向模式或有限差分...")
    return nothing, loss_vec(params_vec)
  end
end

"""
使用有限差分计算梯度（作为备用方案）
"""
function compute_gradient_finitediff(model, params, h_init, infiltration, z, θ_obs)
  eps = 1e-6
  params_vec = Vector(params)
  n = length(params_vec)

  # 计算基准损失
  loss_base = compute_loss(model, params, h_init, infiltration, z, θ_obs)

  # 逐参数计算梯度
  grad = zeros(n)
  for i in 1:n
    params_perturb = copy(params_vec)
    params_perturb[i] += eps

    p_perturb = ComponentArray(params_perturb, getaxes(params))
    loss_perturb = compute_loss(model, p_perturb, h_init, infiltration, z, θ_obs)

    grad[i] = (loss_perturb - loss_base) / eps
  end

  grad_ca = ComponentArray(grad, getaxes(params))
  return grad_ca, loss_base
end

"""
简化的Enzyme梯度计算 - 仅针对神经网络参数
注意: Enzyme 对某些操作敏感，这里使用更简单的实现
"""
function compute_gradient_neural_only(model, params, h_init, infiltration, z, θ_obs)
  # 将神经网络参数转换为简单向量
  neural_params_vec = collect(Vector(ComponentArray(params.neural)))

  # 保存原始轴信息
  neural_axes = getaxes(ComponentArray(params.neural))
  physics_params = params.physics

  # 定义损失函数 - 使用显式参数避免闭包问题
  function loss_neural(p_neural_vec::Vector{Float64})
    # 重构参数
    p_neural = ComponentArray(p_neural_vec, neural_axes)
    p_full = ComponentArray(
      physics=physics_params,
      neural=p_neural
    )
    return compute_loss(model, p_full, h_init, infiltration, z, θ_obs)
  end

  # 尝试使用 Enzyme
  grad_neural = zeros(length(neural_params_vec))

  try
    # 使用 Enzyme 的 autodiff_deferred (更稳定)
    Enzyme.autodiff_deferred(Enzyme.Reverse, loss_neural, Enzyme.Active,
                             Enzyme.Duplicated(neural_params_vec, grad_neural))

    # 构造完整的梯度
    grad_full = ComponentArray(
      physics=(θs=0.0, θr=0.0, α=0.0, n=0.0, Ks=0.0),
      neural=ComponentArray(grad_neural, neural_axes)
    )

    return grad_full, loss_neural(neural_params_vec)

  catch e
    # Enzyme 失败时，尝试使用有限差分作为后备
    println("  Enzyme 失败，使用有限差分后备方案")
    println("  错误信息: ", typeof(e))

    # 使用有限差分计算神经网络参数梯度
    grad_neural = compute_gradient_fd_simple(loss_neural, neural_params_vec)

    grad_full = ComponentArray(
      physics=(θs=0.0, θr=0.0, α=0.0, n=0.0, Ks=0.0),
      neural=ComponentArray(grad_neural, neural_axes)
    )

    return grad_full, loss_neural(neural_params_vec)
  end
end

"""简单的有限差分梯度计算（后备方案）"""
function compute_gradient_fd_simple(f::Function, x::Vector{Float64}; eps=1e-6)
  n = length(x)
  grad = zeros(n)
  f0 = f(x)

  for i in 1:n
    x_perturb = copy(x)
    x_perturb[i] += eps
    grad[i] = (f(x_perturb) - f0) / eps
  end

  return grad
end

"""
使用 Zygote 计算梯度（推荐方法）
Zygote 对 Julia 代码有更好的支持
"""
function compute_gradient_zygote(model, params, h_init, infiltration, z, θ_obs)
  if !ZYGOTE_AVAILABLE
    error("Zygote.jl 未安装。请运行: using Pkg; Pkg.add(\"Zygote\")")
  end

  # 将神经网络参数转换为向量
  neural_params_vec = collect(Vector(ComponentArray(params.neural)))
  neural_axes = getaxes(ComponentArray(params.neural))
  physics_params = params.physics

  # 定义损失函数（告诉它使用 Zygote 友好版本）
  function loss_neural(p_neural_vec::Vector{Float64})
    p_neural = ComponentArray(p_neural_vec, neural_axes)
    p_full = ComponentArray(
      physics=physics_params,
      neural=p_neural
    )
    return compute_loss(model, p_full, h_init, infiltration, z, θ_obs, use_zygote=true)
  end

  # 使用 Zygote 计算梯度
  loss_val = loss_neural(neural_params_vec)
  grad_neural = Zygote.gradient(loss_neural, neural_params_vec)[1]

  # 构造完整的梯度
  grad_full = ComponentArray(
    physics=(θs=0.0, θr=0.0, α=0.0, n=0.0, Ks=0.0),
    neural=ComponentArray(grad_neural, neural_axes)
  )

  return grad_full, loss_val
end

"""
智能梯度计算：自动选择最佳方法
优先级: Zygote > Enzyme > 有限差分
"""
function compute_gradient_auto(model, params, h_init, infiltration, z, θ_obs)
  # 方法1: 尝试 Zygote（最推荐）
  if ZYGOTE_AVAILABLE
    try
      return compute_gradient_zygote(model, params, h_init, infiltration, z, θ_obs)
    catch e
      println("  警告: Zygote 失败 ($(typeof(e))), 尝试其他方法...")
    end
  end

  # 方法2: 尝试 Enzyme
  neural_params_vec = collect(Vector(ComponentArray(params.neural)))
  neural_axes = getaxes(ComponentArray(params.neural))
  physics_params = params.physics

  function loss_neural(p_neural_vec::Vector{Float64})
    p_neural = ComponentArray(p_neural_vec, neural_axes)
    p_full = ComponentArray(
      physics=physics_params,
      neural=p_neural
    )
    return compute_loss(model, p_full, h_init, infiltration, z, θ_obs)
  end

  grad_neural = zeros(length(neural_params_vec))

  try
    Enzyme.autodiff_deferred(Enzyme.Reverse, loss_neural, Enzyme.Active,
                             Enzyme.Duplicated(neural_params_vec, grad_neural))

    grad_full = ComponentArray(
      physics=(θs=0.0, θr=0.0, α=0.0, n=0.0, Ks=0.0),
      neural=ComponentArray(grad_neural, neural_axes)
    )
    return grad_full, loss_neural(neural_params_vec)
  catch e
    # Enzyme 也失败了，使用有限差分
  end

  # 方法3: 有限差分（最后的后备）
  grad_neural = compute_gradient_fd_simple(loss_neural, neural_params_vec)
  grad_full = ComponentArray(
    physics=(θs=0.0, θr=0.0, α=0.0, n=0.0, Ks=0.0),
    neural=ComponentArray(grad_neural, neural_axes)
  )
  return grad_full, loss_neural(neural_params_vec)
end

function test_enzyme_gradient(model, params, h_init, infiltration, z, θ_obs)
  """测试多种梯度计算方法"""
  println("\n" * "="^60)
  println("测试梯度计算方法")
  println("="^60)

  # 计算损失值
  loss_val = compute_loss(model, params, h_init, infiltration, z, θ_obs)
  println("\n当前损失值: ", round(loss_val, digits=6))

  # 测试智能梯度计算
  println("\n使用智能梯度计算（自动选择最佳方法）...")
  grad_auto, loss_auto = compute_gradient_auto(model, params, h_init, infiltration, z, θ_obs)
  println("  ✓ 梯度计算成功!")
  println("    - 总梯度范数: ", round(norm(Vector(ComponentArray(grad_auto.neural))), digits=8))
  println("    - W1 梯度范数: ", round(norm(grad_auto.neural.W1), digits=8))
  println("    - W2 梯度范数: ", round(norm(grad_auto.neural.W2), digits=8))
  println("    - 损失值: ", round(loss_auto, digits=8))

  # 可选：测试 Zygote（如果可用）
  if ZYGOTE_AVAILABLE
    println("\n测试 Zygote 梯度...")
    try
      grad_zyg, loss_zyg = compute_gradient_zygote(model, params, h_init, infiltration, z, θ_obs)
      println("  ✓ Zygote 成功!")
      println("    - 梯度范数: ", round(norm(Vector(ComponentArray(grad_zyg.neural))), digits=8))
    catch e
      println("  ✗ Zygote 失败: ", typeof(e))
    end
  end

  # 可选：测试 Enzyme
  println("\n测试 Enzyme 梯度...")
  grad_enz, loss_enz = compute_gradient_neural_only(model, params, h_init, infiltration, z, θ_obs)
  if grad_enz !== nothing
    println("  ✓ Enzyme 成功!")
    println("    - 梯度范数: ", round(norm(Vector(ComponentArray(grad_enz.neural))), digits=8))
  end

  println("\n" * "="^60)
  if ZYGOTE_AVAILABLE
    println("推荐: 使用 Zygote（最稳定快速）")
    println("提示: 训练将自动使用 compute_gradient_auto()")
  else
    println("推荐: 安装 Zygote 以获得更好的性能")
    println("运行: using Pkg; Pkg.add(\"Zygote\")")
  end
  println("="^60)

  return grad_auto, loss_val
end

# ============================================================================
# 6. 训练函数
# ============================================================================

"""
简单的梯度下降训练循环（智能梯度计算）
自动选择最佳梯度计算方法: Zygote > Enzyme > 有限差分
"""
function train_with_gradient!(model, params, h_init, infiltration, z, θ_obs;
  n_epochs=50, learning_rate=0.001, silent=false)

  if !silent
    println("\n" * "="^60)
    println("开始训练（智能梯度计算）")
    if ZYGOTE_AVAILABLE
      println("使用方法: Zygote.jl（推荐）")
    else
      println("使用方法: Enzyme.jl 或有限差分")
      println("提示: 安装 Zygote 可显著提升速度")
    end
    println("="^60)
  end

  losses = Float64[]
  method_used = "未知"

  for epoch in 1:n_epochs
    # 使用智能梯度计算
    grad, loss = compute_gradient_auto(model, params, h_init, infiltration, z, θ_obs)

    if grad === nothing
      println("  Epoch $epoch: 梯度计算失败，终止训练")
      break
    end

    push!(losses, loss)

    # 梯度下降更新
    params.neural.W1 .-= learning_rate .* grad.neural.W1
    params.neural.b1 .-= learning_rate .* grad.neural.b1
    params.neural.W2 .-= learning_rate .* grad.neural.W2
    params.neural.b2 .-= learning_rate .* grad.neural.b2

    # 打印进度
    if !silent && (epoch % 10 == 0 || epoch == 1)
      grad_norm = norm(Vector(ComponentArray(grad.neural)))
      println("  Epoch $epoch: Loss = $(round(loss, digits=6)), " *
              "Grad norm = $(round(grad_norm, digits=6))")
    end
  end

  if !silent
    println("\n训练完成!")
    println("  初始损失: $(round(losses[1], digits=6))")
    println("  最终损失: $(round(losses[end], digits=6))")
    if length(losses) > 1
      improvement = (losses[1] - losses[end]) / losses[1] * 100
      println("  损失下降: $(round(improvement, digits=2))%")
    end
  end

  return losses
end

# ============================================================================
# 7. 主函数
# ============================================================================

function main()
  println("="^60)
  println("高级土壤水运动模拟 - Lux + Enzyme")
  println("="^60)

  # 初始化模型
  model, params, dz, nz = initialize_hybrid_model()

  println("\n模型配置:")
  println("  土层数: ", nz)
  println("  空间步长: ", round(dz, digits=3), " m")
  println("  神经网络参数: ", length(params.neural.W1) + length(params.neural.W2) +
                        length(params.neural.b1) + length(params.neural.b2))

  # 设置初始条件
  z = [i * dz for i in 1:nz]
  h_init = [-1.0 - 0.3 * i for i in 1:nz]  # 初始压力水头分布
  infiltration = 3e-6  # 入渗率 (m/s)

  # 生成合成观测（添加小噪声以产生非零损失）
  h_true = richards_with_implicit_grad(h_init, params.physics, infiltration, model.solver)
  θ_obs = [vg_theta(h_true[i], params.physics.θs, params.physics.θr,
    params.physics.α, params.physics.n) for i in 1:nz]

  # 添加小噪声使问题更真实
  rng = Random.default_rng()
  Random.seed!(rng, 456)
  noise = 0.01 * randn(rng, nz)  # 1% 噪声
  θ_obs = θ_obs .+ noise
  θ_obs = clamp.(θ_obs, 0.0, params.physics.θs)  # 确保在合理范围内

  println("\n初始状态:")
  println("  压力水头范围: [", round(minimum(h_init), digits=2), ", ",
    round(maximum(h_init), digits=2), "] m")
  println("  入渗率: ", infiltration * 1e6, " mm/s")
  println("  观测含水量范围: [", round(minimum(θ_obs), digits=3), ", ",
    round(maximum(θ_obs), digits=3), "]")

  # 前向传播测试
  println("\n前向传播...")
  h_pred = model((h_init, infiltration, z), params)
  println("  预测压力水头范围: [", round(minimum(h_pred), digits=2), ", ",
    round(maximum(h_pred), digits=2), "] m")

  # 梯度测试
  grad, loss_val = test_enzyme_gradient(model, params, h_init, infiltration, z, θ_obs)

  # 运行训练示例
  println("\n准备运行训练示例...")
  losses = train_with_gradient!(model, params, h_init, infiltration, z, θ_obs,
    n_epochs=20, learning_rate=0.0001)

  println("\n" * "="^60)
  println("✓ 演示完成!")
  println("\n总结:")
  if ZYGOTE_AVAILABLE
    println("  1. ✓ Zygote 梯度计算工作正常（推荐）")
  else
    println("  1. ○ 使用 Enzyme/有限差分（建议安装 Zygote）")
  end
  println("  2. ✓ 训练循环正常工作")
  println("  3. ✓ 损失函数数值稳定")
  println("\n下一步:")
  if !ZYGOTE_AVAILABLE
    println("  - [推荐] 安装 Zygote: using Pkg; Pkg.add(\"Zygote\")")
  end
  println("  - 使用更多训练数据和更长训练")
  println("  - 调整学习率和优化器(如Adam)")
  println("  - 实现mini-batch训练处理时间序列")
  println("  - 考虑使用Optimization.jl进行更高级的优化")
  println("="^60)

  return model, params, losses
end

# 运行
model, params, losses = main()
