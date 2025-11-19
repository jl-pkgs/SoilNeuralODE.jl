#import "@local/modern-cug-report:0.1.3": *
#show: doc => template(doc, footer: "CUG水文气象学2025", header: "")

// =
// ==
=== 1.1.1 方法二：伴随方法（Adjoint Method）

==== 1.1.1.1 核心思想

*关键洞察*：不直接计算 $partial u \/ partial theta$（$n times p$ 维），而是引入*伴随变量* $lambda(t)$（$n$ 维）巧妙避开维度爆炸。

#block(
  fill: rgb("#e6f3ff"),
  inset: 10pt,
  radius: 4pt,
)[
  *直观类比*：伴随方法是连续时间的反向传播

  #table(
    columns: (1fr, 1fr),
    align: (left, left),
    [*标准神经网络*], [*NeuralODE*],
    [前向传播：输入 → 输出], [前向ODE：$u(0) arrow.r u(T)$],
    [反向传播：输出梯度 ← 输入梯度], [反向ODE：$lambda(T) arrow.r lambda(0)$],
  )

  就像backprop是前向传播的"时间反转"，伴随方法是前向ODE的"时间反转"！
]

==== 1.1.1.2 数学推导

*问题设定*：给定ODE $dv(u, t) = f(u, theta, t)$ 和损失 $L = cal(L)(u(T))$，求 $pdv(L, theta)$

*步骤 1*：引入伴随变量，定义伴随变量 $lambda(t)$ 为损失对状态的梯度：
$ lambda(t) = (partial L)/(partial u(t)) $

*步骤 2*：推导伴随方程，通过链式法则和ODE约束，可以推导出 $lambda(t)$ 满足的演化方程：

$ (d lambda)/(d t) = -(partial f)/(partial u)^T lambda, quad lambda(T) = (partial cal(L))/(partial u(T)) $

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *推导*（离散化方法）：状态演化 $u(t + Delta t) approx u(t) + f(u, theta, t) Delta t$

  应用链式法则：
  $
    lambda(t) = pdv(L, u(t)) = pdv(L, u(t + Delta t)) pdv(u(t + Delta t), u(t)) = lambda(t + Delta t) [I + pdv(f, u) Delta t]
  $

  整理后取极限 $Delta t arrow 0$：
  $
    -(d lambda)/(d t) = (pdv(f, u))^T lambda quad arrow.r.double quad (d lambda)/(d t) = -(pdv(f, u))^T lambda
  $ <eq_dlambda>

  #v(-1.2em)
  *关键特性*：
  - *负号*：时间反向（$t: T arrow.r 0$），这是反向传播的本质
  - *转置*：保证梯度正确地反向流动
  - *维度*：$lambda in bb(R)^n$，与 $u$ 同维，避免参数维度爆炸
]

*步骤 3*：计算参数梯度

通过分部积分可以推导出梯度公式：

$ boxed(pdv(L, theta) = integral_0^T lambda(t)^T pdv(f, theta) d t) $

#Blue[推导方法1：]

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *完整推导*（记 $u_theta = pdv(u, theta)$ 为敏感度矩阵）：

  *Step 1*：写出我们想要的量

  根据链式法则，参数梯度为：
  $
    pdv(L, theta) = pdv(L, u(T)) pdv(u(T), theta) = lambda(T)^T u_theta(T)
  $

  *问题*：$u_theta(T)$ 是 $n times p$ 维矩阵，太大！

  #block(
    fill: rgb("#e6f7e6"),
    inset: 8pt,
    radius: 3pt,
  )[
    *推导策略*：我们要把边界项 $lambda(T)^T u_theta(T)$（含高维 $u_theta$）转化成积分形式。

    为什么要积分？
    1. *分散计算*：从一次性计算大矩阵 → 每个时刻计算小的局部导数
    2. *可操纵*：通过选择合适的 $lambda(t)$，消除不想要的高维项
  ]

  #v(0.5em)
  *Step 2*：构造关键恒等式

  考虑乘积求导公式（对任意 $lambda(t)$）：
  $
    dv(, t)[lambda(t)^T u_theta(t)] = dv(lambda^T, t) u_theta + lambda^T dv(u_theta, t)
  $

  #text(fill: rgb("#0066cc"))[
    *为什么对时间积分？*
    - 我们想要的是*终端时刻的值* $lambda(T)^T u_theta(T)$
    - 上面公式只给出了某个时刻 $t$ 的导数
    - 通过*微积分基本定理*，对时间积分可以把边界值和积分项联系起来
  ]

  两边从 0 积分到 $T$（应用微积分基本定理）：
  $
    underbrace([lambda^T u_theta]_0^T, lambda(T)^T u_theta(T) - lambda(0)^T u_theta(0)) = integral_0^T dv(lambda^T, t) u_theta d t + integral_0^T lambda^T dv(u_theta, t) d t
  $

  重排，假设 $u_theta(0) = 0$（初值不依赖参数）：
  $
    lambda(T)^T u_theta(T) = integral_0^T dv(lambda^T, t) u_theta d t + integral_0^T lambda^T dv(u_theta, t) d t
  $

  #v(0.5em)
  *Step 3*：代入敏感度方程

  从 ODE $dv(u, t) = f(u, theta, t)$ 对 $theta$ 求导，得到敏感度方程：
  $
    dv(u_theta, t) = pdv(f, u) u_theta + pdv(f, theta)
  $

  代入 Step 2 的公式：
  $
    lambda(T)^T u_theta(T) = integral_0^T dv(lambda^T, t) u_theta d t + integral_0^T lambda^T [pdv(f, u) u_theta + pdv(f, theta)] d t
  $

  展开：
  $
    lambda(T)^T u_theta(T) = integral_0^T [dv(lambda^T, t) + lambda^T pdv(f, u)] u_theta d t + integral_0^T lambda^T pdv(f, theta) d t
  $

  #v(0.5em)
  *Step 4*：由式#[@eq_dlambda]，$lambda(t)$满足：$dv(lambda^T, t) = -lambda^T pdv(f, u) quad "（伴随方程"$

  那么第一个积分项中的括号变为 0：
  $
    lambda(T)^T u_theta(T) = integral_0^T underbrace([dv(lambda^T, t) + lambda^T pdv(f, u)], = 0) u_theta d t + integral_0^T lambda^T pdv(f, theta) d t
  $

  *高维项被消除了！* 只剩下：
  $ boxed(lambda(T)^T u_theta(T) = integral_0^T lambda^T pdv(f, theta) d t) $
  #v(0.5em)
  *Step 5*：得到最终公式

  结合 Step 1 中 $pdv(L, theta) = lambda(T)^T u_theta(T)$，得到：
  $ boxed(pdv(L, theta) = integral_0^T lambda(t)^T pdv(f, theta) d t) $
]

- *消除敏感度*：通过伴随方程、分部积分巧妙消除高维的 $pdv(u, theta)$
- *只需局部导数*：$pdv(f, theta)$ 是每个时刻的局部雅可比，易于自动微分计算
- *维度独立*：计算复杂度从 $O(n p)$ 降到 $O(n)$，与参数数量 $p$ 无关


#Blue[推导方法2：]

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *推导思路*（记 $u_theta = pdv(u, theta)$ 为敏感度矩阵）：

  1. *问题*：直接计算需要 $u_theta$（$n times p$ 维，太大！）

  2. *技巧*：利用敏感度方程 $dv(u_theta, t) = pdv(f, u) u_theta + pdv(f, theta)$

  3. *分部积分*：对 $integral_0^T lambda^T dv(u_theta, t) d t$ 使用分部积分
    $
      integral_0^T lambda^T dv(u_theta, t) d t = [lambda^T u_theta]_0^T - integral_0^T dv(lambda^T, t) u_theta d t
    $

    利用伴随方程 $dv(lambda^T, t) = -lambda^T pdv(f, u)$，代入得
    $
      = [lambda^T u_theta]_0^T - integral_0^T (-lambda^T pdv(f, u)) u_theta d t = [lambda^T u_theta]_0^T + integral_0^T lambda^T pdv(f, u) u_theta d t
    $

  4. *代入敏感度方程*：
    $
      integral_0^T lambda^T [pdv(f, u) u_theta + pdv(f, theta)] d t = [lambda^T u_theta]_0^T + integral_0^T lambda^T pdv(f, u) u_theta d t
    $

  5. *消除高维项*：两边含 $u_theta$ 的积分项相消，得到
    $
      [lambda^T u_theta]_0^T = integral_0^T lambda^T pdv(f, theta) d t
    $

  6. *应用边界条件*：对终态损失 $L = cal(L)(u(T))$，链式法则给出
    $
      pdv(L, theta) = lambda(T)^T u_theta(T)
    $
    若 $u_theta(0) = 0$（初值不依赖参数），则 $[lambda^T u_theta]_0^T = pdv(L, theta)$

  *结论*：结合5和6，得到最终公式（无需计算 $u_theta$！）
]

*核心洞察*：
- *消除敏感度*：通过分部积分巧妙消除高维的 $pdv(u, theta)$
- *只需局部导数*：$pdv(f, theta)$ 是每个时刻的局部雅可比，易于自动微分计算
- *维度独立*：计算复杂度从 $O(n p)$ 降到 $O(n)$，与参数数量 $p$ 无关

==== 1.1.1.3 计算流程

```julia
# 1. 前向求解 ODE（正常求解，维度 = n）
u_traj = solve(ODEProblem(f, u0, (0, T), θ))  # 对应：du/dt = f(u, θ, t)

# 2. 计算初始伴随变量（在终止时刻）
λ_T = gradient(u -> loss(u), u_traj[end])[1]  # 对应：λ(T) = ∂L/∂u(T)
                                              # 维度 = n
# 3. 反向求解伴随 ODE（维度 = n）
function adjoint_dynamics(λ, p, t)
    u_t = u_traj(t)       # 从前向解插值获取 u(t)
    jac = jacobian(u -> f(u, θ, t), u_t)  # 计算 ∂f/∂u
    return -jac' * λ      # 对应：dλ/dt = -(∂f/∂u)ᵀ λ
end

λ_traj = solve(ODEProblem(adjoint_dynamics, λ_T, (T, 0), θ))
# ↑↑↑ 关键：时间从 T → 0 反向求解 ↑↑↑
```

```julia
# 4. 积分计算参数梯度（维度 = p）
# 对应公式：∂L/∂θ = ∫₀ᵀ λ(t)ᵀ ∂f/∂θ dt

# 定义被积函数
function integrand(t)
    λ_t = λ_traj(t)       # 获取 λ(t)（从反向轨迹插值）
    u_t = u_traj(t)       # 获取 u(t)（从前向轨迹插值）
    ∂f_∂θ = gradient(θ -> f(u_t, θ, t), θ)[1]  # 计算 ∂f/∂θ（局部导数）
    return λ_t' * ∂f_∂θ   # 返回 λ(t)ᵀ ∂f/∂θ
end

# 数值积分（从 0 到 T）
∂L_∂θ = quadgk(integrand, 0, T)[1]
```

#block(
  fill: rgb("#e6f3ff"),
  inset: 10pt,
  radius: 4pt,
)[
  *代码与公式对应关系*：

  #table(
    columns: (1.2fr, 1.5fr, 1.3fr),
    align: (left, left, left),
    [*步骤*], [*公式*], [*代码*],
    [前向ODE], [$dv(u, t) = f(u, theta, t)$], [`u_traj = solve(...)`],
    [边界条件], [$lambda(T) = pdv(L, u(T))$], [`λ_T = gradient(...)`],
    [伴随ODE], [$dv(lambda, t) = -(pdv(f, u))^T lambda$], [`-jac' * λ`],
    [参数梯度], [$pdv(L, theta) = integral_0^T lambda^T pdv(f, theta) d t$], [`quadgk(...)`],
  )

  *关键细节*：
  - 伴随ODE从 $T arrow 0$ 反向求解，但积分从 $0 arrow T$ 正向计算（两者使用插值，方向无关）
  - `jac'` 是雅可比矩阵的转置，对应公式中的 $(pdv(f, u))^T$
  - 最后一步需要数值积分（或在ODE求解过程中累积）
]

#block(
  fill: rgb("#ffe6e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *为什么时间区间是 `(T, 0)` 而不是 `(0, T)`？*

  *❌ 常见错误：用 `λ_0 = gradient(u -> loss(u), u_traj[1])[1]` 计算 λ(0)？*

  *关键区别*：

  - $lambda(T) = pdv(L, u(T))$：$L$ *直接*依赖 $u(T)$，可以直接求导

  - $lambda(0) = pdv(L, u(0))$：$L$ *间接*依赖 $u(0)$，通过链式法则：
    $
      pdv(L, u(0)) = pdv(L, u(T)) pdv(u(T), u(0))
    $

    $pdv(L, u(0))$是要求解的量，而其依赖于另一个未知量$pdv(L, u(T))$，导致问题不可解。
]

==== 1.1.1.4 维度对比

#block(
  fill: rgb("#e6f7e6"),
  inset: 8pt,
  radius: 3pt,
)[
  *符号说明*：
  - $n$：状态变量维度（例如：ODE 有 10 个状态变量）
  - $p$：参数维度（例如：神经网络有 100,000 个参数）
  - $T$：时间长度（与维度分析无关，只影响计算时间）
]

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  [*方法*], [*前向求解*], [*梯度计算*], [*总维度*],
  [正向模式 AD], [$n + n p$], [—], [$n + n p$],
  [伴随方法], [$n$], [$n$], [$2n$],
)

*数值例子*：
- $n = 10$ （状态维度：如 10 维向量 $u in bb(R)^10$）
- $p = 100,000$ （参数维度：如神经网络权重）
- 正向模式：$10 + 10 times 100,000 = 1,000,010$ 维
- 伴随方法：$2 times 10 = 20$ 维
- *加速比*：$50,000 times$！

#block(
  fill: rgb("#fff8e6"),
  inset: 8pt,
  radius: 3pt,
)[
  *具体场景举例*：

  考虑一个简单的物理系统 NeuralODE：

  ```julia
  # 状态：10维向量 [位置x, y, z; 速度vx, vy, vz; 温度T; 压力P; ...]
  u ∈ ℝ¹⁰  # n = 10

  # ODE右端由神经网络定义
  du/dt = NeuralNet(u, θ, t)

  # 神经网络参数：3层，每层 [100, 100, 10]
  θ ∈ ℝ¹⁰⁰'⁰⁰⁰  # p = 100,000（权重+偏置）
  ```

  *正向模式 AD*：需要同时求解 $u(t)$ 和 $pdv(u, theta_i)$ 对所有 $i = 1, dots, 100,000$
  - 实际上是求解 $1 + 100,000 = 100,001$ 个 10 维 ODE 系统！
  - 总维度：$10 times 100,001 approx 1,000,010$

  *伴随方法*：只求解 $u(t)$ 和 $lambda(t)$，都是 10 维
  - 前向求解：$u(t) in bb(R)^10$
  - 反向求解：$lambda(t) in bb(R)^10$
  - 总维度：$10 + 10 = 20$
]

=== 1.1.2 为什么伴随方法可行？

==== 1.1.2.1 关键技巧 1：只计算需要的方向

正向模式：计算*所有*参数方向的导数 $partial u \/ partial theta_i$（$p$ 个方向）

伴随方法：利用"损失是标量"的特性，只计算*一个*方向：$partial L \/ partial u$

#block(
  fill: rgb("#e6f7e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *维度分析*：

  - 正向模式：从 $p$ 个输入（参数）到 $n$ 个输出（状态）
    - 需要 $p$ 次计算（每个参数一次）

  - 反向模式：从 $1$ 个输出（损失）到 $p$ 个输入（参数）
    - 只需 $1$ 次计算（反向传播）

  *通用规则*（自动微分的基本定理）：
  - 输入维度 $p "gg"$ 输出维度 $q$ → 用反向模式
  - 输入维度 $p "ll"$ 输出维度 $q$ → 用正向模式

  NeuralODE：$p = 100,000 "gg" q = 1$ → 反向模式完胜！
]

==== 1.1.2.2 关键技巧 2：时间反转

不直接计算 $partial u(T) \/ partial theta$，而是反向积分伴随变量：

- 前向：$t: 0 arrow.r T$，计算 $u(t)$
- 反向：$t: T arrow.r 0$，计算 $lambda(t)$

这样避免了存储整个 $(partial u \/ partial theta)(t)$ 的轨迹（会占用 $O(n p T)$ 内存）。

=== 1.1.3 与普通反向传播的类比

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  [*特性*], [*标准神经网络*], [*NeuralODE*],
  [前向], [离散层 $x^((l+1)) = f(x^((l)))$], [连续 ODE $d u \/ d t = f(u, t)$],
  [反向], [离散反向传播 $delta^((l))$], [连续伴随 ODE $d lambda \/ d t$],
  [梯度], [链式法则], [伴随积分],
  [复杂度], [$O(L)$（$L$ = 层数）], [$O(T)$（$T$ = 时间长度）],
)

*相同点*：都是反向传播的思想

*不同点*：NeuralODE 是*连续*时间的反向传播

=== 1.1.4 总结

#block(
  fill: rgb("#e6f3ff"),
  inset: 10pt,
  radius: 4pt,
)[
  *伴随方法的三大优势*：

  + *维度独立*：复杂度 $O(n)$，与参数数量 $p$ 无关
  + *内存高效*：不需要存储 $partial u \/ partial theta$ 的完整轨迹
  + *计算高效*：与前向求解 ODE 的成本相当

  *代价*：

  - 需要存储前向解的插值信息（InterpolatingAdjoint）
  - 或者重新计算前向解（BacksolveAdjoint）

  *结论*：对于高维参数系统（神经网络），伴随方法是*唯一*可行的选择！
]

== 1.2 InterpolatingAdjoint 详解

=== 1.2.1 三种伴随方法对比

SciMLSensitivity.jl 提供多种伴随算法：

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, center, center),
  [*算法*], [*策略*], [*内存*], [*稳定性*],
  [`BacksolveAdjoint`], [反向求解 ODE], [$O(1)$], [差（刚性问题）],
  [`InterpolatingAdjoint`], [前向插值 + 反向求解], [$O(n)$], [好],
  [`QuadratureAdjoint`], [自适应积分], [$O(n)$], [最好],
)

=== 1.2.2 BacksolveAdjoint 的问题

#block(
  fill: rgb("#e6f7e6"),
  inset: 8pt,
  radius: 3pt,
)[
  *核心区别*：BacksolveAdjoint 和 InterpolatingAdjoint 的本质差异

  #table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    [*特性*], [*BacksolveAdjoint*], [*InterpolatingAdjoint*],
    [前向求解], [求解 $u(t)$，*不保存*轨迹], [求解 $u(t)$，*保存插值信息*],
    [内存消耗], [$O(1)$（最小）], [$O(n)$（中等）],
    [反向求解时\需要 $u(t)$], [*重新反向求解*前向ODE], [从保存的插值中直接获取],
    [数值稳定性], [差（反向求解可能不稳定）], [好（使用前向解）],
  )

  *关键点*：你的理解完全正确！InterpolatingAdjoint 就是保留了每一步的 u(t)（通过插值），而 BacksolveAdjoint 没有保存。
]

*反向求解*的数值不稳定性问题：

```julia
# 前向求解（两种方法都一样）
u_forward = solve(ODEProblem(f, u0, (0, T), θ))

# ===== 反向求解伴随 ODE（需要用到 u(t)）=====

# BacksolveAdjoint 的做法：
function adjoint_rhs_backsolve(λ, p, t)
    # ❌ 问题：需要 u(t)，但没有保存！
    # 解决：重新从 T 反向求解到 t（反向求解前向ODE！）
    u_t = solve(ODEProblem(f, u_final, (T, t), θ))[end]  # 反向求解
    jac = jacobian(f, u_t, θ, t)
    return -jac' * λ
end

# InterpolatingAdjoint 的做法：
function adjoint_rhs_interpolating(λ, p, t)
    # ✓ 直接从保存的插值中获取 u(t)
    u_t = u_forward(t)  # 插值（使用前向求解的结果）
    jac = jacobian(f, u_t, θ, t)
    return -jac' * λ
end
```

#block(
  fill: rgb("#ffe6e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *BacksolveAdjoint 的陷阱*：

  为什么反向求解前向 ODE 会不稳定？

  考虑一个简单例子：$dv(u, t) = -100 u$（稳定的指数衰减）

  - *正向求解*（$t: 0 arrow T$）：
    $
      u(t) = u_0 e^(-100 t) quad arrow.r quad "指数衰减（稳定）"
    $

  - *反向求解*（$t: T arrow 0$，令 $tau = T - t$）：
    $
      dv(u, tau) = 100 u quad arrow.r quad u(tau) = u_T e^(100 tau) quad arrow.r quad "指数增长（不稳定！）"
    $

  *结果*：
  - 前向稳定的系统，反向求解变成了指数增长
  - 数值误差被放大 $e^(100 tau)$ 倍
  - 对于刚性（stiff）系统，这个问题尤其严重

  神经网络经常产生刚性系统 → `BacksolveAdjoint` 容易失败
]

=== 1.2.3 InterpolatingAdjoint 的解决方案

*核心机制*：使用前向解的插值，避免反向求解状态

```julia
# 1. 前向求解（保存插值信息）
u_interp = solve(ODEProblem(f, u0, (0, T), θ), save_everystep=true)

# 2. 反向求解伴随变量（但使用前向解的插值）
function adjoint_rhs(λ, p, t)
    u_t = u_interp(t)  # 插值获取 u(t)，而非反向求解
    jac = jacobian(f, u_t, θ, t)
    return -jac' * λ
end

λ = solve(ODEProblem(adjoint_rhs, λT, (T, 0), θ))
```

*关键优势*：
- 伴随变量 $lambda$ 反向求解，但*状态* $u(t)$ 使用前向插值
- 数值稳定性大幅提升（因为 $u(t)$ 不涉及反向积分）
- 内存开销适中（存储前向解的插值系数）

=== 1.2.4 内存与计算的权衡

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  [*方法*], [*内存开销*], [*计算开销*],
  [`BacksolveAdjoint`], [$O(1)$ — 最低], [1× 前向求解 + 1× 反向求解],
  [`InterpolatingAdjoint`], [$O(n)$ — 中等], [1× 前向求解 + 1× 反向求解],
  [`QuadratureAdjoint`], [$O(n^2)$ — 最高], [最少（自适应积分）],
)

*Checkpointing 优化*：

```julia
# 使用 checkpointing 减少内存
sensealg = InterpolatingAdjoint(checkpointing=true)
```

- 只在检查点存储完整状态
- 区间内重新前向求解（用计算换内存）


#pagebreak()
== 1.3 autojacvec 与 ZygoteVJP

=== 1.3.1 什么是 Vector-Jacobian Product (VJP)？

伴随方法需要频繁计算*向量-Jacobian 积*：

$ v^T J = v^T (partial f)/(partial u) $

其中 $v = lambda$（伴随变量）。

*直接计算的问题*：

```julia
# 朴素方法：先计算整个 Jacobian，再乘向量
J = jacobian(f, u, θ, t)  # O(n²) 存储
result = v' * J            # O(n²) 计算
```

对于神经网络（$n$ 可能很大），存储和计算 $J$ 都不现实。

=== 1.3.2 高效 VJP 计算

*关键洞察*：我们只需要 $v^T J$ 这个*向量*，不需要完整的 $J$ 矩阵！

自动微分可以直接计算 VJP，无需构造 $J$：

```julia
# 使用自动微分计算 VJP（无需显式构造 J）
vjp = gradient(u -> dot(v, f(u, θ, t)), u)[1]
```

- 计算复杂度：$O(n)$
- 内存复杂度：$O(n)$

=== 1.3.3 autojacvec 参数选项

`autojacvec` 指定如何计算 VJP：

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  [*选项*], [*方法*], [*适用场景*],
  [`nothing`], [自动选择], [默认（通常选 `true`）],
  [`false`], [有限差分（FiniteDiff.jl）], [函数不可微],
  [`true`], [正向模式 AD（ForwardDiff.jl）], [小规模系统],
  [`ZygoteVJP()`], [反向模式 AD（Zygote.jl）], [神经网络（向量化代码）],
)

=== 1.3.4 为什么 NeuralODE 用 ZygoteVJP？

*Zygote 的优势*：

```julia
# 神经网络是向量化的、out-of-place 的
y = model(x, θ)  # 矩阵乘法、激活函数等

# Zygote 对这类代码高度优化
∇ = Zygote.gradient(x -> dot(v, model(x, θ)), x)[1]
```

+ *反向模式 AD*：适合高维输入、低维输出（VJP 正是如此）
+ *向量化操作*：GPU 友好，利用 BLAS/cuBLAS
+ *纯函数式*：out-of-place 代码无副作用，Zygote 易于追踪

#block(
  fill: rgb("#fff4e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *性能陷阱*：

  Zygote 对 in-place 操作支持不佳：

  ```julia
  # 慢（Zygote.Buffer 开销大）
  function f!(du, u, p, t)
      du[1] = ...
  end

  # 快（纯函数式，Zygote 优化良好）
  function f(u, p, t)
      return [...]
  end
  ```

  这也是为什么 NeuralODE 使用 `ODEFunction{false}` (out-of-place)！
]

=== 1.3.5 ForwardDiff vs Zygote

#table(
  columns: (auto, auto, auto),
  align: (left, center, center),
  [*特性*], [*ForwardDiff*], [*Zygote*],
  [模式], [正向模式], [反向模式],
  [适用场景], [低维输入], [高维输入],
  [VJP 计算], [$O(n)$ 次前向传播], [1 次反向传播],
  [JVP 计算], [1 次前向传播], [$O(n)$ 次反向传播],
  [神经网络], [慢], [快],
)

*VJP 计算对比*：

```julia
# ForwardDiff（正向模式）：需要 n 次前向传播
vjp_fd = [ForwardDiff.derivative(ε -> dot(v, f(u + ε*eᵢ)), 0) for i in 1:n]

# Zygote（反向模式）：1 次反向传播
vjp_zygote = Zygote.gradient(u -> dot(v, f(u)), u)[1]
```

对于神经网络，Zygote 快得多！

== 1.4 完整工作流程

=== 1.4.1 NeuralODE 的前向与反向传播

```julia
# 前向传播
function forward(x, θ)
    dudt(u, p, t) = neural_net(u, p)
    prob = ODEProblem(dudt, x, (0, T), θ)
    sol = solve(prob, Tsit5();
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return sol[end]
end

# 反向传播（自动由 InterpolatingAdjoint 处理）
∇θ = gradient(θ -> loss(forward(x, θ)), θ)[1]
```

*内部发生的事情*：

+ *前向求解*：
  - 求解 ODE：$d u \/ d t = "NN"(u, theta)$
  - 保存解的插值信息

+ *反向传播*：
  - 计算 $lambda(T) = partial cal(L) \/ partial u(T)$
  - 反向求解伴随 ODE：
    $ (d lambda)/(d t) = -(partial "NN")/(partial u)^T lambda $
  - 使用前向插值获取 $u(t)$
  - 使用 Zygote 计算 $(partial "NN") \/ (partial u)^T lambda$
  - 积分得到 $partial L \/ partial theta$

=== 1.4.2 关键参数汇总

```julia
solve(prob, alg;
    sensealg = InterpolatingAdjoint(
        autojacvec = ZygoteVJP(),  # 使用 Zygote 计算 VJP
        checkpointing = false       # 是否使用 checkpointing
    ),
    save_everystep = false          # 是否保存所有步（训练时通常不需要）
)
```

== 1.5 何时使用 InterpolatingAdjoint？

#block(
  fill: rgb("#e6f7e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *推荐场景*：

  + ✓ *NeuralODE*（默认选择）
  + ✓ *刚性问题*（BacksolveAdjoint 不稳定）
  + ✓ *中等规模问题*（内存不是主要瓶颈）
  + ✓ *向量化神经网络*（配合 ZygoteVJP）

  *不推荐场景*：

  + ✗ *极大规模问题*（内存受限）→ 用 `BacksolveAdjoint` + checkpointing
  + ✗ *非刚性、非常简单的问题*（内存敏感）→ 用 `BacksolveAdjoint`
  + ✗ *极高精度要求*（数值误差敏感）→ 用 `QuadratureAdjoint`
]

== 1.6 性能对比示例

```julia
using DifferentialEquations, SciMLSensitivity, BenchmarkTools, Zygote

# 简单 NeuralODE
model(u, p) = p[1] * tanh.(p[2] .* u)

function test_sensealg(sensealg)
    prob = ODEProblem((u,p,t) -> model(u,p), [1.0], (0.0, 1.0), [0.5, 2.0])

    ∇p = gradient([0.5, 2.0]) do p
        sol = solve(prob, Tsit5(), p=p, sensealg=sensealg)
        sum(sol[end])
    end

    return ∇p[1]
end

# 性能对比
@btime test_sensealg(BacksolveAdjoint())        # ~50 μs（但可能不稳定）
@btime test_sensealg(InterpolatingAdjoint(autojacvec=ZygoteVJP()))  # ~80 μs（稳定）
@btime test_sensealg(QuadratureAdjoint())       # ~120 μs（最稳定）
```

*权衡*：
- `BacksolveAdjoint`：最快，但刚性问题不稳定
- `InterpolatingAdjoint`：平衡性能与稳定性（*NeuralODE 默认*）
- `QuadratureAdjoint`：最稳定，但略慢

== 1.7 总结

#block(
  fill: rgb("#e6f3ff"),
  inset: 10pt,
  radius: 4pt,
)[
  *InterpolatingAdjoint + ZygoteVJP 的设计智慧*：

  + *伴随方法*：将 $O(n p)$ 的梯度计算降到 $O(n)$
  + *插值策略*：避免数值不稳定的反向求解
  + *Zygote VJP*：高效计算向量-Jacobian 积，适配神经网络
  + *out-of-place*：配合 Zygote 优化，实现高性能

  这一系列设计使得 NeuralODE 的训练既*高效*又*稳定*，是 SciML 生态的核心技术之一。
]

#pagebreak()
== 1.8 伴随方法的适用范围

=== 1.8.1 能否用于非 ODE 形式？

#block(
  fill: rgb("#e6f7e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *问题*：伴随方法能用于 `st = f(x, st, p)` 这样的形式吗？

  这个形式与标准 ODE $dv(u, t) = f(u, theta, t)$ 不同，需要分情况讨论：
]

==== 情况 1：隐式方程（Implicit Equation）

如果 `st = f(x, st, p)` 是一个*隐式方程*（需要求解 st）：

```julia
# 例如：st = tanh(W*x + U*st + b)  （隐式）
# 需要求解：st - tanh(W*x + U*st + b) = 0
```

*答案*：*可以用，但需要修改！* → 使用*隐式函数定理*的伴随方法

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *关键思路*（隐式函数定理）：

  定义残差 $G(x, s, p) = s - f(x, s, p) = 0$

  求解 $pdv(L, p)$ 时，利用隐式函数定理：
  $
    pdv(s, p) = -[pdv(G, s)]^(-1) pdv(G, p)
  $

  引入伴随变量 $lambda$，通过求解线性系统避免显式计算 $pdv(s, p)$：
  $
    [pdv(G, s)]^T lambda = pdv(L, s) \
    pdv(L, p) = -lambda^T pdv(G, p)
  $

  *优势*：同样避免了高维矩阵求逆！
]

==== 情况 2：递归更新（Recursive Update）

如果是时间递归：`st[n+1] = f(x[n], st[n], p)`

```julia
# 例如：RNN 的递归
# s₀ = initial_state
# s₁ = f(x₁, s₀, p)
# s₂ = f(x₂, s₁, p)
# ...
# sₜ = f(xₜ, sₜ₋₁, p)
```

*答案*：*可以用！* → 这就是*反向传播通过时间（BPTT）*，伴随方法的离散版本

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *RNN 的伴随方法*（反向传播）：

  *前向传播*：
  $
    s_(t+1) = f(x_t, s_t, p), quad t = 0, 1, dots, T-1
  $

  *定义伴随变量*：
  $
    lambda_t = pdv(L, s_t)
  $

  *反向传播*（从 T 到 0）：
  $
    lambda_t = lambda_(t+1) pdv(f, s_t) + pdv(L_t, s_t)
  $

  *参数梯度*：
  $
    pdv(L, p) = sum_(t=0)^(T-1) lambda_(t+1) pdv(f, p)
  $

  *对比*：

  #table(
    columns: (1fr, 1fr),
    align: (left, left),
    [*NeuralODE（连续）*], [*RNN（离散）*],
    [$dv(u, t) = f(u, theta, t)$], [$s_(t+1) = f(x_t, s_t, p)$],
    [伴随 ODE：$dv(lambda, t) = -(pdv(f, u))^T lambda$], [BPTT：$lambda_t = lambda_(t+1) pdv(f, s_t)$],
    [积分：$integral_0^T lambda^T pdv(f, theta) d t$], [求和：$sum_t lambda_(t+1) pdv(f, p)$],
  )

  *结论*：RNN 的反向传播 = 伴随方法的离散化版本！
]

==== 情况 3：依赖额外输入 x

如果 ODE 还依赖时变输入 $x(t)$：

```julia
# du/dt = f(u, x(t), θ, t)
# 例如：du/dt = NN(u, x(t), θ)
```

*答案*：*完全可以用！* 伴随方法不受影响

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *修改*：只需在计算局部导数时包含 $x(t)$

  *伴随方程*（不变）：
  $
    dv(lambda, t) = -(pdv(f, u))^T lambda
  $

  *参数梯度*（额外项）：
  $
    pdv(L, theta) = integral_0^T lambda(t)^T pdv(f, theta) d t
  $

  *对输入 x 的梯度*（如果需要）：
  $
    pdv(L, x(t)) = lambda(t)^T pdv(f, x)
  $

  *示例*：Neural Controlled Differential Equations (NCDEs)
  ```julia
  dudt(u, p, t) = NN(u, x_interp(t), p)  # x_interp(t) 是时变输入
  # 伴随方法照常使用，InterpolatingAdjoint 会自动处理
  ```
]

=== 1.8.2 总结：伴随方法的适用性

#table(
  columns: (auto, auto, 1fr),
  align: (left, center, left),
  [*问题类型*], [*能用吗*], [*方法*],
  [ODE：$dv(u, t) = f(u, theta, t)$], [✓], [标准伴随方法],
  [带输入的 ODE：$dv(u, t) = f(u, x(t), theta, t)$], [✓], [标准伴随方法（计算导数时包含 x）],
  [隐式方程：$s = f(x, s, p)$], [✓], [隐式函数定理 + 伴随方法],
  [递归更新：$s_(t+1) = f(x_t, s_t, p)$], [✓], [反向传播（BPTT）= 离散伴随方法],
  [DAE：$0 = g(u, dv(u, t), t)$], [✓], [DAE 伴随方法（更复杂）],
  [优化问题：$min_u L(u, p)$], [✓], [拉格朗日乘子法（伴随方法的推广）],
)

*核心原则*：只要有"链式求导"，就可以设计伴随方法！

关键是：
1. 定义合适的伴随变量
2. 推导伴随方程（消除高维敏感度矩阵）
3. 反向传播梯度信息

#pagebreak()
== 1.9 NeuralODE 的实现细节

=== 1.9.1 solve 函数的返回值

在 DiffEqFlux.jl 的 NeuralODE 实现中：

```julia
function (n::NeuralODE)(x, p, st)
    model = StatefulLuxLayer{fixed_state_type(n.model)}(n.model, nothing, st)

    dudt(u, p, t) = model(u, p)
    ff = ODEFunction{false}(dudt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)

    return (
        solve(prob, n.args...;
            sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
            n.kwargs...),
        model.st)
end
```

#block(
  fill: rgb("#e6f3ff"),
  inset: 10pt,
  radius: 4pt,
)[
  *`solve` 返回什么？*

  `solve` 返回一个 `ODESolution` 对象，包含以下信息：

  #table(
    columns: (auto, 1fr),
    align: (left, left),
    [*属性*], [*说明*],
    [`sol.u`], [状态轨迹：`Vector{Vector}`，保存求解过程中各时间点的状态],
    [`sol.t`], [时间点：`Vector{Float64}`，对应 `sol.u` 的时间],
    [`sol[end]`], [终态：$u(T)$，等价于 `sol.u[end]`],
    [`sol(t)`], [插值函数：可以查询任意时刻 $t$ 的状态 $u(t)$],
    [`sol.alg`], [使用的求解器算法],
    [`sol.prob`], [原始的 ODE 问题],
    [`sol.retcode`], [求解状态码（成功/失败）],
  )

  *NeuralODE 的返回值*：元组 `(ODESolution, state)`

  ```julia
  return (solve(...), model.st)
  #        ↑            ↑
  #        ODE解        Lux模型的状态（如BatchNorm的running mean）
  ```
]

=== 1.9.2 ODESolution 的使用

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *常见用法*：

  ```julia
  # 1. 获取终态（最常用）
  sol, st = neural_ode(x0, p, st)
  u_final = sol[end]  # 或 sol.u[end]

  # 2. 获取完整轨迹
  u_trajectory = sol.u  # Vector{Vector{Float64}}
  t_points = sol.t      # Vector{Float64}

  # 3. 插值查询任意时刻
  u_at_05 = sol(0.5)    # 获取 t=0.5 时的状态

  # 4. 计算损失（通常只用终态）
  loss = mse(sol[end], target)

  # 5. 可视化轨迹
  using Plots
  plot(sol, vars=(1,2,3), label="State trajectory")
  # 或
  plot(sol.t, [u[1] for u in sol.u], label="u₁(t)")
  ```

  *为什么返回元组 `(sol, st)`？*

  - `sol`：ODE 的求解结果
  - `st`：Lux 模型的状态（stateful layers 如 BatchNorm、Dropout）
  - 分离设计：ODE 求解器不管理模型状态，由 Lux 框架管理
]

=== 1.9.3 训练时如何使用

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *典型训练循环*：

  ```julia
  using DiffEqFlux, Lux, Optimization, OptimizationOptimisers

  # 定义 NeuralODE
  model = Lux.Chain(Dense(2, 50, tanh), Dense(50, 2))
  neural_ode = NeuralODE(model, (0.0, 1.0), Tsit5())

  # 初始化参数和状态
  ps, st = Lux.setup(rng, neural_ode)

  # 损失函数
  function loss_function(ps, st)
      # 前向传播
      sol, st_new = neural_ode(x0, ps, st)

      # 只用终态计算损失
      pred = sol[end]  # u(T)
      loss = sum(abs2, pred - target)

      return loss, st_new
  end

  # 训练
  opt = Adam(0.01)
  for epoch in 1:1000
      loss_val, grads, st = gradient(loss_function, ps, st)
      ps = Optimisers.update!(opt, ps, grads)

      if epoch % 100 == 0
          println("Epoch $epoch: Loss = $loss_val")
      end
  end

  # 预测
  sol_final, _ = neural_ode(x0_test, ps, st)
  prediction = sol[end]
  ```

  *关键点*：
  - `sol[end]` 是最常用的，因为损失通常定义在终态
  - `sensealg = InterpolatingAdjoint(...)` 自动处理反向传播
  - 状态 `st` 需要在训练循环中更新（虽然对于纯前馈网络通常不变）
]

=== 1.9.4 save_everystep 参数

#block(
  fill: rgb("#fff4e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *控制内存使用*：

  ```julia
  # 保存所有步（默认，用于可视化）
  sol = solve(prob, Tsit5(), save_everystep=true)
  # sol.u 包含所有时间点的状态
  # 内存：O(n × 时间步数)

  # 只保存起点和终点（训练时推荐）
  sol = solve(prob, Tsit5(), save_everystep=false)
  # sol.u 只包含 [u(0), u(T)]
  # 内存：O(n)
  ```

  *训练时的选择*：

  #table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    [*场景*], [*save_everystep*], [*原因*],
    [训练（只用终态损失）], [`false`], [节省内存，只需 $u(T)$],
    [训练（多时刻损失）], [`true`], [需要 $u(t_1), u(t_2), dots$],
    [推理/可视化], [`true`], [需要完整轨迹画图],
  )

  *InterpolatingAdjoint 的内存*：

  即使 `save_everystep=false`，`InterpolatingAdjoint` 仍会保存插值信息（用于反向传播）
  - 保存的是插值系数，不是所有时间步的完整状态
  - 内存消耗：$O(n)$，不是 $O(n times "步数")$
]

=== 1.9.5 完整示例

```julia
using DiffEqFlux, Lux, DifferentialEquations, Plots, Random

# 1. 定义神经网络
rng = Random.default_rng()
model = Lux.Chain(Dense(2, 50, tanh), Dense(50, 2))

# 2. 创建 NeuralODE
tspan = (0.0, 1.0)
neural_ode = NeuralODE(model, tspan, Tsit5())

# 3. 初始化
ps, st = Lux.setup(rng, neural_ode)

# 4. 前向传播
x0 = [1.0, 0.0]
sol, st = neural_ode(x0, ps, st)

# 5. 使用 ODESolution
println("Initial state: ", sol[1])      # u(0)
println("Final state: ", sol[end])      # u(T)
println("State at t=0.5: ", sol(0.5))   # u(0.5) 通过插值

# 6. 可视化
plot(sol, vars=(1,2), label="Neural ODE trajectory",
     xlabel="u₁", ylabel="u₂")

# 7. 训练（损失定义在终态）
target = [0.0, 1.0]
function loss_fn(ps, st)
    sol, st_new = neural_ode(x0, ps, st)
    loss = sum(abs2, sol[end] - target)
    return loss, st_new
end

# 梯度自动处理伴随方法
loss_val, grads, st = gradient(loss_fn, ps, st)
```

=== 1.9.6 sensealg vs alg：两种不同的算法

#block(
  fill: rgb("#ffe6e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *常见误解*：`sensealg` 是求解 ODE 的算法吗？

  *答案*：*不是！* 这是两种完全不同的算法：
]

#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  [*参数*], [*全称*], [*作用*],
  [`alg`], [Algorithm（求解器算法）], [*前向求解* ODE 的数值方法],
  [`sensealg`], [*Sensitivity* Algorithm], [*反向传播* 计算梯度的方法],
)

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *详细对比*：

  ```julia
  solve(prob,
        Tsit5(),  # ← alg: 前向求解 ODE 的算法
        sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()))
        #          ↑ sensealg: 反向传播计算梯度的算法
  ```

  #table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    [*特性*], [*alg*（求解器）], [*sensealg*（敏感度算法）],
    [*全称*], [ODE Solver Algorithm], [Sensitivity Algorithm],
    [*用途*], [前向求解 ODE], [反向传播计算梯度],
    [*作用阶段*], [前向传播], [反向传播（自动微分）],
    [*常见选项*], [`Tsit5()`, `Rodas5()`, `VCABM()`], [`InterpolatingAdjoint`, `BacksolveAdjoint`],
    [*影响什么*], [ODE 求解的精度、速度], [梯度计算的方法、效率],
    [*必需吗*], [可选（有默认值）], [可选（有默认值）],
  )
]

=== 1.9.7 sensealg 的命名来源

#block(
  fill: rgb("#e6f3ff"),
  inset: 10pt,
  radius: 4pt,
)[
  *为什么叫 "sensitivity algorithm"？*

  *sensitivity* 在数值计算中指*灵敏度分析*或*敏感度分析*：

  $ "sensitivity" = pdv(u(T), theta) $

  研究"输出 $u(T)$ 对参数 $theta$ 的敏感程度"

  *历史背景*：

  - 在控制论、最优化领域，计算 $pdv(u, theta)$ 的方法统称为 *sensitivity analysis*
  - 不同的计算方法（前向模式、伴随方法等）称为 *sensitivity algorithms*
  - SciML 继承了这个术语

  *对应关系*：

  #table(
    columns: (1fr, 1fr),
    align: (left, left),
    [*机器学习术语*], [*SciML/控制论术语*],
    [反向传播（Backpropagation）], [伴随方法（Adjoint Method）],
    [梯度计算], [敏感度分析（Sensitivity Analysis）],
    [自动微分算法], [敏感度算法（Sensitivity Algorithm）],
  )
]

=== 1.9.8 可用的 sensealg 选项

#block(
  fill: rgb("#fff8e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *SciMLSensitivity.jl 提供的方法*：

  #table(
    columns: (auto, 1fr, auto),
    align: (left, left, center),
    [*sensealg*], [*说明*], [*推荐*],
    [`InterpolatingAdjoint()`], [伴随方法 + 插值（默认）], [✓✓✓],
    [`BacksolveAdjoint()`], [伴随方法 + 反向求解], [✓],
    [`QuadratureAdjoint()`], [伴随方法 + 自适应积分], [✓✓],
    [`ForwardDiffSensitivity()`], [正向模式自动微分], [✗],
    [`ReverseDiffAdjoint()`], [ReverseDiff.jl], [✓],
    [`TrackerAdjoint()`], [Tracker.jl（已过时）], [✗],
    [`ZygoteAdjoint()`], [Zygote.jl（试验性）], [✓],
  )

  *选择建议*：

  - *NeuralODE*：`InterpolatingAdjoint(autojacvec=ZygoteVJP())` ✓✓✓
  - *刚性问题*：`InterpolatingAdjoint()` 或 `QuadratureAdjoint()` ✓✓
  - *内存受限*：`BacksolveAdjoint()` ✓
  - *小参数量*：`ForwardDiffSensitivity()` ✓
]

=== 1.9.9 完整示例：两种算法的配合

```julia
using DifferentialEquations, SciMLSensitivity

# 定义 ODE
function f(u, p, t)
    return p[1] * u
end

u0 = [1.0]
tspan = (0.0, 1.0)
p = [0.5]

prob = ODEProblem(f, u0, tspan, p)

# ========== 前向求解 ==========
# alg 控制前向求解
sol = solve(prob,
            Tsit5(),        # ← alg: 使用 Tsitouras 5/4 Runge-Kutta 方法
            saveat=0.1)

# ========== 反向传播（计算梯度）==========
using Zygote

function loss(p)
    sol = solve(prob, Tsit5(), p=p,
                sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()))
                #          ↑ sensealg: 使用伴随方法计算梯度
    return sum(abs2, sol[end] - [2.0])
end

# 梯度计算自动使用 sensealg 指定的方法
dp = gradient(loss, p)[1]

println("Gradient: ", dp)
```

#block(
  fill: rgb("#e6f7e6"),
  inset: 10pt,
  radius: 4pt,
)[
  *执行流程*：

  1. *前向传播*（`alg` 起作用）：
     - 使用 `Tsit5()` 求解 ODE：$dv(u,t) = f(u,p,t)$
     - 得到 $u(T)$

  2. *反向传播*（`sensealg` 起作用）：
     - `gradient(loss, p)` 触发自动微分
     - 使用 `InterpolatingAdjoint` 计算 $pdv(L, p)$
     - 利用伴随方法避免高维矩阵

  *关键*：两个算法在不同阶段工作，互不干扰！
]

=== 1.9.10 总结

#block(
  fill: rgb("#e6f3ff"),
  inset: 10pt,
  radius: 4pt,
)[
  *记忆要点*：

  - `alg` = 求解器算法 = *前向*求解 ODE
  - `sensealg` = 敏感度算法 = *反向*计算梯度

  *命名来源*：
  - `sens-` = sensitivity（敏感度/灵敏度）
  - 来自控制论/数值优化领域的术语
  - 机器学习中等价于"自动微分算法"或"反向传播方法"

  *类比*：
  ```
  solve(prob, alg,     sensealg)
           ↓           ↓
        "怎么走"    "怎么算梯度"
           ↓           ↓
        前向传播    反向传播
  ```
]
