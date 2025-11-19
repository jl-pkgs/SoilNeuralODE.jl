#import "@local/modern-cug-report:0.1.3": *
#show: doc => template(doc, footer: "Enzyme AD 运行时活动分析", header: "")

= Enzyme 自动微分中的运行时活动问题分析

== 1. 问题现象

在使用 Enzyme.jl 对物理信息神经网络进行自动微分时，遇到以下错误：

```
ERROR: EnzymeRuntimeActivityError: Detected potential need for runtime activity.
```

错误提示需要使用 `set_runtime_activity(Reverse)` 来启用运行时活动分析。

== 2. 根本原因

=== 2.1 什么是"活动"(Activity)

在自动微分中，*活动* 指的是某个变量是否需要计算梯度：

- *Active*：变量依赖于可微分参数，需要计算梯度
- *Const*：变量是常量，不需要梯度
- *Duplicated*：变量及其梯度 shadow buffer

=== 2.2 静态分析的局限性

Enzyme 默认使用*静态活动分析*：在编译时确定哪些内存需要微分。但当代码混合常量和可微分变量时，静态分析可能失败。

== 3. 触发问题的代码模式

=== 3.1 案例 1：向量混合运算

```julia
# hybrid_forward 函数第 73 行
θ_hybrid = θ_physics .+ 0.01f0 .* corrections
```

*问题分析：*

#table(
  columns: (auto, auto, auto),
  align: (left, center, left),
  [*变量*], [*来源*], [*活动状态*],
  [`θ_physics`], [物理计算], [不含 NN 参数（静态分析认为是常量）],
  [`corrections`], [神经网络输出], [依赖 `nn_params`（需要微分）],
  [`θ_hybrid`], [混合结果], [*不确定！*（静态分析失败）],
)

*Enzyme 的困惑：*

```julia
// Enzyme 编译时的思考过程
if θ_physics 是常量 && corrections 需要微分:
    θ_hybrid 应该是什么？  // ❌ 静态分析无法确定
```

=== 3.2 案例 2：边界约束

```julia
# hybrid_forward 函数第 81 行
θ_hybrid = clamp.(θ_hybrid, soil_params[2] + 0.01f0, soil_params[1] - 0.01f0)
#                            ↑ 常量边界               ↑ 常量边界
#                  ↑ 可微分向量
```

- `θ_hybrid` 需要微分
- `soil_params[2]`, `soil_params[1]` 是常量
- `clamp` 操作混合了两者

=== 3.3 案例 3：物理约束损失

```julia
# loss_func 函数第 98 行
loss_physics = sum(max.(0.0f0, θ_pred .- θ_s) .^ 2)
#                       ↑常量   ↑可微分  ↑常量
```

- `θ_pred` 依赖 `nn_params`（需要微分）
- `0.0f0` 和 `θ_s` 是常量
- `max` 操作产生条件分支，增加分析难度

== 4. 安全的代码模式

以下代码*不会*触发问题：

```julia
# loss_func 函数第 100 行
return loss_data + 0.1f0 * loss_physics
```

*为什么安全？*

- `loss_data` 和 `loss_physics` 都是*标量*
- 标量运算简单，Enzyme 容易处理
- 即使 `0.1f0` 是常量，标量乘法也不影响活动分析

== 5. 解决方案对比

=== 5.1 方案 A：启用运行时活动（推荐）

```julia
Enzyme.autodiff(
  set_runtime_activity(Reverse),  # ✓ 启用运行时分析
  loss_func,
  Active,
  Duplicated(ps, dps),
  ...
)
```

*优点：*
- 简单快速，一行代码解决
- 保证正确性
- 适用于复杂混合运算

*缺点：*
- 轻微性能损失（通常可忽略）

=== 5.2 方案 B：重写代码避免混合

```julia
# 原始代码（有问题）
θ_hybrid = θ_physics .+ 0.01f0 .* corrections

# 重写方案（避免混合）
function add_correction(θ_phys, corr, scale)
  # 明确标记所有操作都需要微分
  return θ_phys .+ scale .* corr
end
θ_hybrid = add_correction(θ_physics, corrections, 0.01f0)
```

*优点：*
- 最优性能

*缺点：*
- 复杂度高，需要重构大量代码
- 不总是可行（某些物理模型难以避免混合）

== 6. 运行时活动的工作原理

=== 6.1 静态分析 vs 运行时分析

#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  [*特性*], [*静态分析*], [*运行时分析*],
  [时机], [编译时], [执行时],
  [速度], [快（一次分析）], [稍慢（每次执行检查）],
  [准确性], [保守（可能失败）], [精确（动态跟踪）],
  [适用场景], [简单、纯函数式代码], [复杂、混合运算代码],
)

=== 6.2 运行时分析示例

```julia
# 运行时 Enzyme 的动态决策
function runtime_check(θ_physics, corrections):
  if has_gradient(corrections):  # 运行时检查
    mark_active(θ_physics .+ corrections)  # ✓ 正确标记
  else:
    mark_const(θ_physics .+ corrections)
```

== 7. Enzyme autodiff 参数详解

=== 7.1 完整签名

```julia
Enzyme.autodiff(
  mode,           # 微分模式
  func,           # 目标函数
  return_type,    # 返回值类型标注
  arg1_annotation,# 参数 1 的活动标注
  arg2_annotation,# 参数 2 的活动标注
  ...
)
```

=== 7.2 返回值类型标注

#table(
  columns: (auto, auto, 1fr),
  align: (left, center, left),
  [*标注*], [*适用*], [*说明*],
  [`Active`], [标量返回], [损失函数、能量函数等],
  [`Duplicated(ret, dret)`], [向量/矩阵返回], [需要完整梯度的复杂返回值],
  [`Const`], [常量返回], [不需要微分的返回值],
)

*关键规则：*

#box-red[
  *标量返回值*必须用 `Active`，否则报错：
  
  ```
  ERROR: Duplicated Returns not yet handled
  ```
]

=== 7.3 参数标注

```julia
# 示例
Enzyme.autodiff(
  Reverse,
  loss_func,
  Active,                    # 返回标量
  Duplicated(ps, dps),       # 参数：需要梯度
  Const(θ_init),             # 常量：初始状态
  Const(depths),             # 常量：深度向量
  Const(nn_model),           # 常量：模型结构
  Const(st),                 # 常量：模型状态
  ...
)
```

== 8. 实践建议

=== 8.1 开发流程

#box-blue[
  *阶段 1：快速原型*
  
  1. 先用 `set_runtime_activity(Reverse)`
  2. 确保代码正确性
  3. 完成功能开发
  
  *阶段 2：性能优化*（可选）
  
  1. Profiling 找出性能瓶颈
  2. 如果运行时活动是瓶颈，考虑重写
  3. 通常情况下性能差异很小，无需优化
]

=== 8.2 调试技巧

*问题：如何定位触发问题的具体代码行？*

```julia
# 查看完整堆栈
try
  Enzyme.autodiff(...)
catch err
  println(code_typed(err))  # 显示 LLVM IR
end
```

*问题：如何验证梯度正确性？*

```julia
# 与 Zygote 对比
using Zygote

# Enzyme 梯度
dps_enzyme = make_zero(ps)
Enzyme.autodiff(set_runtime_activity(Reverse), loss_func, Active, 
                Duplicated(ps, dps_enzyme), ...)

# Zygote 梯度
dps_zygote = Zygote.gradient(p -> loss_func(p, ...), ps)[1]

# 对比
@assert isapprox(dps_enzyme, dps_zygote, rtol=1e-5)
```

== 9. 常见错误与解决

=== 9.1 错误 1：Duplicated Returns

```
ERROR: Duplicated Returns not yet handled
```

*原因：*损失函数返回标量，但使用了 `Duplicated`

*解决：*改用 `Active`

```julia
# ❌ 错误
Enzyme.autodiff(Reverse, loss_func, Duplicated(ps, dps), ...)

# ✓ 正确
Enzyme.autodiff(Reverse, loss_func, Active, Duplicated(ps, dps), ...)
```

=== 9.2 错误 2：Runtime Activity

```
ERROR: EnzymeRuntimeActivityError
```

*原因：*混合常量和可微分变量

*解决：*启用运行时活动

```julia
# ❌ 错误
Enzyme.autodiff(Reverse, ...)

# ✓ 正确
Enzyme.autodiff(set_runtime_activity(Reverse), ...)
```

=== 9.3 错误 3：LLVM Type Mismatch

```
Mismatched activity for: ...
```

*原因：*参数标注错误（如把需要微分的参数标记为 `Const`）

*解决：*检查所有参数标注

```julia
# 确保可微分参数用 Duplicated
Duplicated(ps, dps)  # ps 依赖于优化目标

# 确保常量参数用 Const
Const(θ_init)        # 初始条件不需要微分
```

== 10. 总结

#table(
  columns: (auto, 1fr),
  align: (left, left),
  [*核心概念*], [运行时活动分析是 Enzyme 处理复杂混合运算的机制],
  [*适用场景*], [物理信息神经网络、混合模型、约束优化],
  [*性能影响*], [轻微（通常 < 5%），可忽略],
  [*推荐做法*], [开发阶段始终启用，生产阶段根据 Profiling 决定],
  [*关键要点*], [标量返回用 `Active`，向量返回用 `Duplicated`],
)

#box-green[
  *最佳实践：*
  
  对于物理信息神经网络等复杂应用，*始终使用*：
  
  ```julia
  Enzyme.autodiff(
    set_runtime_activity(Reverse),
    loss_func,
    Active,
    Duplicated(params, dparams),
    ...
  )
  ```
  
  除非 Profiling 证明性能是瓶颈，否则无需优化。
]
