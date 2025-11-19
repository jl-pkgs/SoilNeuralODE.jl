#import "@local/modern-cug-report:0.1.3": *
#show: doc => template(doc, footer: "CUG水文气象学2025", header: "")


= 1 SciMLSensitivity 中的 InterpolatingAdjoint

在 NeuralODE 的实现中，有一行关键的反向传播配置：

```julia
solve(prob, n.args...;
  sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
  n.kwargs...)
```

这涉及到微分方程的灵敏度分析（sensitivity analysis），即如何高效计算参数梯度。

== 1.1 伴随方法（Adjoint Method）基础

=== 1.1.1 问题设定：NeuralODE 的训练目标

考虑一个简单的 NeuralODE 训练任务：

```julia
# 神经网络定义导数
dudt(u, θ, t) = neural_net(u, θ)

# 求解 ODE
u_solution = solve(ODEProblem(dudt, u0, (0, T), θ))

# 计算损失（例如：预测误差）
loss_value = mse(u_solution[end], target)
```

*训练目标*：找到最优参数 $theta^*$ 使得损失最小

$ theta^* = arg min_theta L(theta) $

其中损失函数依赖于 ODE 的解：

$ L(theta) = cal(L)(u(T; theta)) $

$ dv(u, t) = f(u, theta, t), quad u(0) = u_0 $

#block(
  fill: rgb("#fff4e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *关键理解*：参数 $theta$ 如何影响损失？

  这是一个*隐式依赖*的链条：

  $ theta arrow.r.long f(u, theta, t) arrow.r.long u(t; theta) arrow.r.long u(T; theta) arrow.r.long L(theta) $

  + *步骤 1*：$theta$ 出现在右侧函数 $f(u, theta, t)$ 中
    - 例如：神经网络的权重和偏置

  + *步骤 2*：$f$ 决定了 ODE 的解 $u(t; theta)$
    - 不同的 $theta$ → 不同的导数函数 → 不同的解轨迹

  + *步骤 3*：终止时刻的解 $u(T; theta)$ 被用于计算损失
    - 例如：与目标值 $u_"target"$ 的误差

  + *步骤 4*：损失 $L(theta)$ 最终依赖于 $theta$
    - 虽然 $theta$ 不直接出现在 $cal(L)(dot)$ 中，但通过 $u(T; theta)$ 隐式依赖

  *数学表达*：
  $ L(theta) = underbrace(cal(L), "损失函数") (underbrace(u(T; theta), "ODE 解")) $

  其中 $u(T; theta)$ 是通过求解 ODE 得到的，而 ODE 的右侧包含 $theta$。

  *具体例子*：

  ```julia
  # 神经网络（参数为 θ）
  f(u, θ, t) = tanh(θ₁ * u + θ₂)

  # 求解 ODE（解依赖于 θ）
  u_solution(θ) = solve(ODEProblem((u,p,t) -> f(u,θ,t), u0, tspan))

  # 计算损失（通过 u(T;θ) 依赖于 θ）
  L(θ) = (u_solution(θ)[end] - target)²

  # 改变 θ → 改变 f → 改变 u(t) → 改变 L
  ```

  这就是为什么需要计算 $partial L \/ partial theta$：找到让损失最小的 $theta$！
]

#block(
  fill: rgb("#f0f0f0"),
  inset: 10pt,
  radius: 4pt,
)[
  *ODE 公式解读*：$dv(u, t) = f(u, theta, t), quad u(0) = u_0$

  + *$u(t)$*：系统的状态（state），是时间 $t$ 的函数
    - 例如：物理位置 $(x, y, z)$、浓度、温度等
    - 在 NeuralODE 中：特征向量，$u in bb(R)^n$

  + *$display(dv(u, t))$*：状态对时间的导数（变化率）
    - 表示"状态如何随时间演化"
    - 类比：速度是位置的导数

  + *$f(u, theta, t)$*：右侧函数（RHS, right-hand side）
    - 输入：当前状态 $u$，参数 $theta$，时间 $t$
    - 输出：状态的瞬时变化率
    - 在 NeuralODE 中：$f$ 是一个神经网络！

  + *$theta$*：模型参数
    - 传统 ODE：物理参数（扩散系数、反应速率等）
    - NeuralODE：神经网络的权重和偏置

  + *$u(0) = u_0$*：初始条件（initial condition）
    - 在时刻 $t = 0$ 时，状态的值
    - 求解 ODE 的起点

  *完整含义*：
  "从初始状态 $u_0$ 出发，状态 $u(t)$ 按照神经网络 $f$ 定义的规则随时间演化。"
]

*核心挑战*：如何计算梯度 $partial L \/ partial theta$？

==== 1.1.1.1 NeuralODE 的具体例子

```julia
# 定义神经网络
model = Chain(
    Dense(2, 50, tanh),
    Dense(50, 2)
)

# 右侧函数：f(u, θ, t)
function dudt(u, θ, t)
    return model(u, θ)  # 神经网络定义导数
end

# 初始状态
u0 = [1.0, 0.0]  # 例如：初始位置

# 求解 ODE
tspan = (0.0, 10.0)  # 时间范围
prob = ODEProblem(dudt, u0, tspan, θ)
sol = solve(prob)

# sol(t) 给出任意时刻 t 的状态 u(t)
```

*物理意义*：
- 如果 $u = [x, y]$ 是粒子位置
- 那么 $d u \/ d t = [v_x, v_y]$ 是速度
- 神经网络学习的是"速度场" → 自动发现粒子的运动规律

*对比传统 ODE*：

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  [*特性*], [*传统 ODE*], [*NeuralODE*],
  [右侧函数], [手写公式（例如 $-k u$）], [神经网络 `model(u, θ)`],
  [参数], [物理常数（$k, D, alpha$ 等）], [神经网络权重（数千到数百万）],
  [设计方式], [基于领域知识], [数据驱动学习],
  [优势], [可解释性强], [表达能力强，自动发现规律],
)

=== 1.1.2 方法一：正向模式自动微分（为什么不可行）

==== 1.1.2.1 正向模式AD的工作原理

正向模式自动微分的思路：同时追踪*值*和*导数*。

*简单例子*：

```julia
# 计算 f(x) = x² + 2x
x = 3.0
dx_dx = 1.0  # ∂x/∂x = 1

# 前向传播：同时计算值和导数
y1 = x * x           # y1 = 9
dy1_dx = 2 * x * dx_dx  # ∂(x²)/∂x = 2x = 6

y2 = 2 * x           # y2 = 6
dy2_dx = 2 * dx_dx   # ∂(2x)/∂x = 2

y = y1 + y2          # y = 15
dy_dx = dy1_dx + dy2_dx  # ∂y/∂x = 8
```

==== 1.1.2.2 应用到 ODE：灾难性的维度爆炸

对于 ODE，正向模式需要*同时*求解原始方程和导数方程：

$ dv(u, t) = f(u, theta, t) $

$
  pdv(, theta) (dv(u, t)) = (d)/(d t) ((partial u)/(partial theta)) = (partial f)/(partial u) (partial u)/(partial theta) + (partial f)/(partial theta)
$

#block(
  fill: rgb("#e6f3ff"),
  inset: 10pt,
  radius: 4pt,
)[
  *导数方程的推导*：如何得到第二个公式？

  *目标*：计算 $display((partial u(t))/(partial theta))$（状态对参数的导数）

  *步骤 1*：从原始 ODE 开始
  $ dv(u, t) = f(u, theta, t) $

  *步骤 2*：两边同时对 $theta$ 求偏导
  $ pdv(, theta) (dv(u, t)) = pdv(, theta) [f(u, theta, t)] $

  *步骤 3*：左边——交换求导顺序
  $ pdv(, theta) (dv(u, t)) = (d)/(d t) ((partial u)/(partial theta)) $

  （交换偏导和时间导数的顺序，这在满足一定条件时是允许的）

  *步骤 4*：右边——使用多元函数的链式法则
  $
    pdv(, theta) [f(u, theta, t)] = underbrace((partial f)/(partial u), n times n "矩阵") underbrace((partial u)/(partial theta), n times p "矩阵") + underbrace((partial f)/(partial theta), n times p "矩阵")
  $

  - 第一项：$f$ 通过 $u$ 间接依赖 $theta$（$theta arrow.r u arrow.r f$）
  - 第二项：$f$ 直接依赖 $theta$

  *步骤 5*：合并
  $
    (d)/(d t) ((partial u)/(partial theta)) = (partial f)/(partial u) (partial u)/(partial theta) + (partial f)/(partial theta)
  $

  *解释*：
  - 左边：敏感度矩阵 $display((partial u)/(partial theta))$ 的时间演化
  - 右边：由两部分驱动
    + 间接影响：$display((partial f)/(partial u))$ 描述 $u$ 的变化如何影响导数
    + 直接影响：$display((partial f)/(partial theta))$ 描述 $theta$ 如何直接影响导数
]

==== 1.1.2.3 维度分析：为什么是灾难

假设：
- $u in bb(R)^n$（状态维度）
- $theta in bb(R)^p$（参数维度）

那么 $display((partial u)/(partial theta))$ 是什么？

#block(
  fill: rgb("#fff4e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *雅可比矩阵*：$display((partial u)/(partial theta) in bb(R)^(n times p))$

  $
    (partial u)/(partial theta) = mat(
      (partial u_1)/(partial theta_1), (partial u_1)/(partial theta_2), dots.h, (partial u_1)/(partial theta_p);
      (partial u_2)/(partial theta_1), (partial u_2)/(partial theta_2), dots.h, (partial u_2)/(partial theta_p);
      dots.v, dots.v, dots.down, dots.v;
      (partial u_n)/(partial theta_1), (partial u_n)/(partial theta_2), dots.h, (partial u_n)/(partial theta_p);
    )
  $

  *含义*：矩阵的第 $(i, j)$ 元素是"状态的第 $i$ 个分量对参数的第 $j$ 个分量的敏感度"。
]

*具体例子*：

```julia
# 小规模例子
n = 2  # 状态：u = [u₁, u₂]
p = 3  # 参数：θ = [θ₁, θ₂, θ₃]

# ∂u/∂θ 是 2×3 矩阵
∂u_∂θ = [
    ∂u₁/∂θ₁  ∂u₁/∂θ₂  ∂u₁/∂θ₃
    ∂u₂/∂θ₁  ∂u₂/∂θ₂  ∂u₂/∂θ₃
]
```

==== 1.1.2.4 什么是增广系统（Augmented System）？

#block(
  fill: rgb("#e6f3ff"),
  inset: 10pt,
  radius: 4pt,
)[
  *增广系统*：将原始 ODE 和导数 ODE 合并成一个更大的联合系统。

  *原始问题*：只求解状态 $u(t)$
  $ dv(u, t) = f(u, theta, t) $

  *增广问题*：同时求解状态 $u(t)$ 和敏感度 $display((partial u)/(partial theta))$
  $
    dv(, t) mat(u; (partial u)/(partial theta)) = mat(
      f(u, theta, t);
      (partial f)/(partial u) (partial u)/(partial theta) + (partial f)/(partial theta)
    )
  $

  把两个方程"堆叠"成一个大系统 → "增广"（augment）
]

*具体例子*：

假设简单的 ODE：
$ dv(u, t) = theta u $

其中 $u in bb(R)^2$（2维状态），$theta in bb(R)^3$（3个参数）。

*原始系统*（2维）：
$ dv(, t) mat(u_1; u_2) = mat(theta_1 u_1; theta_2 u_2) $

*导数系统*（2×3 = 6维）：
$
  dv(, t) mat(
    (partial u_1)/(partial theta_1), (partial u_1)/(partial theta_2), (partial u_1)/(partial theta_3);
    (partial u_2)/(partial theta_1), (partial u_2)/(partial theta_2), (partial u_2)/(partial theta_3)
  ) = mat(
    u_1 + theta_1 (partial u_1)/(partial theta_1), theta_1 (partial u_1)/(partial theta_2), theta_1 (partial u_1)/(partial theta_3);
    theta_2 (partial u_2)/(partial theta_1), u_2 + theta_2 (partial u_2)/(partial theta_2), theta_2 (partial u_2)/(partial theta_3)
  )
$

*增广系统*（2 + 6 = 8维）：
```julia
# 状态向量被"增广"为
z = [
    u₁,           # 原始状态
    u₂,           # 原始状态
    ∂u₁/∂θ₁,      # 导数
    ∂u₁/∂θ₂,      # 导数
    ∂u₁/∂θ₃,      # 导数
    ∂u₂/∂θ₁,      # 导数
    ∂u₂/∂θ₂,      # 导数
    ∂u₂/∂θ₃       # 导数
]

# 求解增广 ODE
dz/dt = augmented_rhs(z, θ, t)
```

正向模式需要同时求解：
+ *原始 ODE*（$n$ 维）：$display(dv(u, t) = f(u, theta, t))$
+ *导数 ODE*（$n times p$ 维）：$display((d)/(d t) ((partial u)/(partial theta)) = ...)$

*总维度*：$n + n p$

==== 1.1.2.5 为什么叫"增广"？

#table(
  columns: (auto, auto, auto),
  align: (left, center, left),
  [*术语*], [*维度*], [*含义*],
  [原始系统], [$n$], [只包含原始状态变量],
  [导数系统], [$n p$], [包含所有敏感度],
  [*增广系统*], [$n + n p$], [原始 + 导数（增强/扩充）],
)

"增广"（augment）= 扩充、增强，即把原始系统"扩充"成包含导数信息的大系统。

==== 1.1.2.6 为什么会产生增广系统？（不是我们想要的！）

#block(
  fill: rgb("#fff4e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *关键理解*：增广系统不是"选择"，而是正向模式AD的*必然代价*。

  *问题根源*：正向模式AD的工作原理

  回顾正向模式的基本思想：
  ```julia
  # 计算函数值的同时，追踪导数
  x = 3.0
  dx_dθ = 1.0  # 种子：∂x/∂θ = 1

  y = f(x)
  dy_dθ = f'(x) * dx_dθ  # 同时计算导数
  ```

  *应用到 ODE*：

  + *时刻 $t = 0$*：
    - 已知 $u(0) = u_0$
    - 已知 $display((partial u)/(partial theta)(0) = (partial u_0)/(partial theta))$（通常是 0）

  + *时刻 $t = t_1$*：
    - 要计算 $u(t_1)$，需要求解 $display(dv(u, t) = f(u, theta, t))$
    - 要计算 $display((partial u)/(partial theta)(t_1))$，需要求解 $display((d)/(d t)((partial u)/(partial theta)) = ...)$

  + *关键矛盾*：
    - 导数方程 $display((d)/(d t)((partial u)/(partial theta)))$ 依赖于 $u(t)$
    - 无法"先求解 $u(t)$，再求解导数"
    - 必须*同时*求解两者 → 增广系统

  *为什么不能分开求解？*

  因为导数方程中包含 $u(t)$：
  $
    (d)/(d t) ((partial u)/(partial theta)) = underbrace((partial f)/(partial u)(u(t), theta, t), "依赖 u(t)") (partial u)/(partial theta) + (partial f)/(partial theta)
  $

  如果先求 $u(t)$，再求导数，那就不是"自动"微分了，而是"数值微分"（有限差分）！
]

==== 1.1.2.7 正向模式 vs 其他方法

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  [*方法*], [*如何计算*], [*精度*],
  [数值微分（有限差分）], [先求 $u(theta)$，再用 $(u(theta+epsilon) - u(theta))\/epsilon$], [低（截断误差）],
  [正向模式AD], [同时求解 $u$ 和 $partial u \/ partial theta$（增广系统）], [高（机器精度）],
  [伴随方法], [先求 $u$，再反向求解伴随 ODE], [高（机器精度）],
)

*正向模式的优势*：精度高（机器精度）

*正向模式的代价*：增广系统维度爆炸

*伴随方法的智慧*：既保持高精度，又避免增广系统！

==== 1.1.2.8 可视化理解：为什么"绑定"在一起

*正向模式的"束缚"*：

```julia
# 时间步进循环（必须同时更新）
for t in [t₀, t₁, t₂, ..., T]
    # 步骤 1：更新状态
    u[t+Δt] = u[t] + Δt * f(u[t], θ, t)

    # 步骤 2：更新导数（依赖刚才的 u[t]！）
    ∂u_∂θ[t+Δt] = ∂u_∂θ[t] + Δt * (
        ∂f_∂u(u[t], θ, t) * ∂u_∂θ[t] +  # 需要当前的 u[t]
        ∂f_∂θ(u[t], θ, t)                # 需要当前的 u[t]
    )
end
```

因为 $display((partial u)/(partial theta)[t+Delta t])$ 的计算需要当前时刻的 $u[t]$，所以必须"绑定"在一起求解 → 增广系统。

*伴随方法的"解耦"*：

```julia
# 步骤 1：前向求解（只求 u，保存轨迹）
for t in [t₀, t₁, t₂, ..., T]
    u[t+Δt] = u[t] + Δt * f(u[t], θ, t)
    save(u[t])  # 保存以备后用
end

# 步骤 2：反向求解（只求 λ，使用已保存的 u[t]）
for t in [T, T-Δt, ..., t₀]
    u_t = load(u[t])  # 从保存的轨迹读取
    λ[t-Δt] = λ[t] - Δt * (∂f_∂u(u_t, θ, t))' * λ[t]
end
```

两个过程*解耦*了！各自只需要 $n$ 维，维度不再爆炸。

#block(
  fill: rgb("#ffe6e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *维度灾难*：

  假设：
  - $u in bb(R)^n$（状态维度）
  - $theta in bb(R)^p$（参数维度）

  那么 $partial u \/ partial theta in bb(R)^(n times p)$（雅可比矩阵）

  *增广系统的维度*：
  $ n + n times p $

  *具体数值*：
  - NeuralODE：$n = 10$（状态），$p = 100,000$（参数）
  - 增广系统：$10 + 10 times 100,000 = 1,000,010$ 维！
  - 对于百万参数的神经网络 → *完全不可行*
]

==== 1.1.2.9 数值例子

```julia
# 小规模例子
n = 3           # 状态维度（例如：x, y, z 坐标）
p = 1000        # 参数数量（小型神经网络）

# 正向模式需要求解的系统维度
dim_forward = n + n * p
# = 3 + 3 × 1000 = 3003 维

# 对于每个时间步，需要计算 3003 维的导数！
# 求解器的计算量 ∝ dim³ ≈ (3000)³ ≈ 27,000,000,000 操作
```

对比普通神经网络训练（反向传播）：只需要前向传播一次（$n$ 维），然后反向传播一次（$n$ 维）。
