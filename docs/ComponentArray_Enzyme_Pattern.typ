#import "@preview/modern-cug-report:0.1.3": *
#show: doc => template(doc, footer: "CUG水文气象学2025", header: "")
 
// #import "@preview/codly:0.2.0": *
// #show: codly-init.with()

= ComponentArray with Enzyme and SciMLSensitivity

在使用 `Enzyme.jl` 结合 `SciMLSensitivity.jl` 对 ODE 进行微分时，直接将 `ComponentArray` 作为参数 `p` 传递给 `ODEProblem` 可能会导致梯度为零。这是因为灵敏度分析工具（如 `EnzymeVJP`）在内部处理参数时可能会进行解包或重建，从而破坏了 Enzyme 对 `ComponentArray` 结构的追踪。

为了解决这个问题，我们采用了 **"Unwrap / Rewrap" (解包/重包)** 模式。

== 核心思想

1.  **Unwrap (解包)**: 在构建 `ODEProblem` 之前，将 `ComponentArray` 转换为普通的 `Vector` (`p_flat`)，并保存其轴信息 (`p_axes`)。将平铺的 `Vector` 传递给求解器和 Enzyme。
2.  **Rewrap (重包)**: 在 ODE 系统函数内部，利用传入的 `Vector` 和 `p_axes` 重建 `ComponentArray` 视图。这样既能保证求解器和 AD 看到的是标准的数组，又能让物理代码享受命名参数的便利。

== 代码实现

=== 1. Setup (Unwrap)

首先，创建 `ComponentArray`，然后分离出数据向量和轴。

```julia
using ComponentArrays

# 定义参数
p_true = ComponentArray(θ_s=0.45, θ_r=0.05, Ks=10.0, α=0.01, n=2.0)

# Unwrap: 获取轴和纯向量
p_axes = getaxes(p_true)
p_flat = Vector(p_true)

# 使用 p_flat 初始化 ODEProblem
u0 = fill(0.2, 20)
prob = ODE.ODEProblem((dθ, θ, p, t) -> richards_eq_inner!(dθ, θ, p, t, soil, p_axes), u0, tspan, p_flat)
```

=== 2. ODE Function (Rewrap)

在 ODE 函数中，使用 `ComponentArray(p_vec, p_axes)` 重建参数对象。

```julia
function richards_eq_inner!(dθ, θ, p_vec, t, soil, p_axes)
  # Rewrap: 重建 ComponentArray 视图
  p = ComponentArray(p_vec, p_axes)
  
  # 现在可以使用命名访问参数
  Ks = p.Ks
  α = p.α
  # ... 物理计算 ...
end
```

=== 3. Adjoint / Loss Function

在定义损失函数时，确保闭包捕获了 `p_axes`，并且 `remake` 使用的是向量形式的参数。

```julia
function loss_adjoint_enzyme(u0, p_vec, soil, prob, p_axes, alg=ODE.Tsit5())
  sensealg = SMS.InterpolatingAdjoint(autojacvec=SMS.EnzymeVJP())
  
  # 闭包捕获 p_axes
  f_closure = (dθ, θ, p, t) -> richards_eq_inner!(dθ, θ, p, t, soil, p_axes)
  
  # Remake 使用 p_vec
  new_prob = ODE.remake(prob, f=f_closure, u0=u0, p=p_vec)
  
  sol = ODE.solve(new_prob, alg; saveat=0.1, sensealg=sensealg)
  return sum(sum(sol))
end
```

=== 4. Enzyme Call

最后，在调用 `Enzyme.autodiff` 时，对 `p_flat` 进行微分 (`Duplicated`)，而 `p_axes` 作为常量 (`Const`) 传递。

```julia
dp_flat = make_zero(p_flat)

Enzyme.autodiff(
  Enzyme.Reverse,
  loss_adjoint_enzyme,
  Enzyme.Active,
  Enzyme.Duplicated(u0, du0),
  Enzyme.Duplicated(p_flat, dp_flat), # 对向量微分
  Enzyme.Duplicated(soil_struct, dsoil),
  Enzyme.Const(prob),
  Enzyme.Const(p_axes) # 轴是常量
)

# 如果需要，可以将梯度转换回 ComponentArray
dp_grad = ComponentArray(dp_flat, p_axes)
```
