#import "@local/modern-cug-report:0.1.3": *
#show: doc => template(doc, footer: "CUG水文气象学2025", header: "")

= Enzyme.jl 使用注意事项

#align(center)[
  _基于 Enzyme.jl 官方文档 FAQ 整理_
]

== 概述

Enzyme.jl 是一个高性能的自动微分库，基于 LLVM 编译器实现。本文档总结了在使用 Enzyme.jl 时需要特别注意的事项和常见问题。

== 1. 常见函数支持情况

=== 1.1 条件函数 (`min`, `max`, `clamp`)

*支持情况：* 基本支持，但存在特殊情况。

*注意事项：*

- 当涉及 `Inf`（无穷大）或 `NaN` 时，可能产生意外的梯度结果
- 中间计算中的 `Inf` 值可能被传播，导致非零梯度

*示例问题：*
```julia
f(y) = min(1.0, y)
# 当 y = Inf 时，可能得到非零梯度
```

*解决方案：*
启用"强零模式"（Strong Zero Mode）来抑制无穷值：

```julia
Enzyme.API.strictAliasing!(false)
Enzyme.API.maxtypeoffset!(64)
# 启用强零模式
Enzyme.API.set_strong_zero(true)
```

=== 1.2 其他数学函数

*支持良好的函数：*
- 基本算术：`+`, `-`, `*`, `/`, `^`
- 三角函数：`sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- 指数对数：`exp`, `log`, `log10`, `sqrt`
- 双曲函数：`sinh`, `cosh`, `tanh`

*需要注意的函数：*
- `abs`：在零点不可微
- `sign`：几乎处处导数为零
- 取整函数（`floor`, `ceil`, `round`）：导数为零或未定义

== 2. 控制流支持

=== 2.1 条件语句 (`if-else`)

*基本支持：* ✓ 支持

*关键问题：* **运行时活性错误** (`EnzymeRuntimeActivityError`)

*问题原因：*
当变量的"活性"（是否需要梯度）在运行时才能确定时，Enzyme 无法在编译时做出决策。

*示例：*
```julia
function conditional_compute(x, flag)
    if flag  # flag 的值在运行时确定
        return x^2
    else
        return 2*x
    end
end
```

*解决方案：*

1. *启用运行时活性模式：*
```julia
Enzyme.API.runtimeActivity!(true)
```

2. *重写代码避免条件活性：*
```julia
# 不好：条件依赖于可能活跃的变量
if x > 0
    result = expensive_computation(x)
else
    result = 0.0
end

# 更好：使用三元运算符或数学形式
result = (x > 0) ? expensive_computation(x) : 0.0
# 或使用 max/min
result = max(0.0, expensive_computation(x))
```

=== 2.2 循环 (`for`, `while`)

*基本支持：* ✓ 支持

*注意事项：*

1. *固定次数循环更安全：*
```julia
# 好：循环次数固定
for i in 1:10
    x = x + f(i)
end

# 需要注意：循环次数依赖于输入
for i in 1:round(Int, x)  # x 是活跃变量
    # ...
end
```

2. *循环中的累积变量：*
```julia
# 确保累积变量正确初始化
result = 0.0  # 明确类型
for i in 1:n
    result += compute(x[i])
end
```

== 3. 数组操作限制

=== 3.1 原地修改

*关键规则：* 原地修改的数组必须使用 `Duplicated` 标注。

*常见错误：*
```julia
# ❌ 错误：临时数组标记为 Const
function bad_example(x)
    temp = zeros(10)  # 这个会被原地修改
    for i in 1:10
        temp[i] = x^i
    end
    return sum(temp)
end

# 调用
autodiff(Reverse, bad_example, Active, Active(2.0))
# 可能得到错误的梯度！
```

*正确做法：*
```julia
# ✓ 正确：使用 Duplicated 传递临时数组
function good_example!(result, x)
    for i in 1:10
        result[i] = x^i
    end
    return sum(result)
end

# 调用
temp = zeros(10)
dtemp = zeros(10)
autodiff(Reverse, good_example!,
         Duplicated(temp, dtemp), Active(2.0))
```

=== 3.2 数组重塑 (`reshape`)

*问题：* `reshape` 可能导致 Enzyme 无法追踪导数。

*示例问题：*
```julia
function use_reshape(W, x)
    W_matrix = reshape(W, 10, 5)  # 重塑向量为矩阵
    return W_matrix * x
end
```

*可能的错误：* `EnzymeMutabilityException`

*解决方案：*

1. *避免在热路径中使用 `reshape`：*
```julia
# 不好
function forward(params, x)
    W = reshape(params.W, hidden, input)  # 每次都 reshape
    return W * x
end

# 更好：预先重塑或使用固定形状
struct Params
    W::Matrix{Float64}  # 直接存储为矩阵
end
```

2. *使用视图代替复制：*
```julia
# 如果必须重塑，考虑使用 view（但 Enzyme 支持有限）
W_matrix = view(W, :)  # 仅在某些情况下有效
```

=== 3.3 稀疏数组

*特殊问题：* `SparseMatrixCSC` 会自动丢弃显式零值。

*问题示例：*
```julia
using SparseArrays
A = sparse([1.0, 2.0, 0.0])
dA = similar(A)  # ❌ 结构可能不正确
```

*解决方案：*
使用 `Enzyme.make_zero()` 创建正确的零初始化影子：

```julia
A = sparse([1.0, 2.0, 0.0])
dA = Enzyme.make_zero(A)  # ✓ 正确的零结构
```

== 4. 类型和可变性要求

=== 4.1 `Duplicated` 类型匹配

*规则：* 原始值和导数必须具有相同类型。

```julia
# ❌ 错误：类型不匹配
x = Float64[1.0, 2.0]
dx = Float32[0.0, 0.0]  # 类型不同！
autodiff(Reverse, f, Duplicated(x, dx))
# 错误：类型不匹配

# ✓ 正确：类型相同
x = Float64[1.0, 2.0]
dx = Float64[0.0, 0.0]
autodiff(Reverse, f, Duplicated(x, dx))
```

*原因：* 确保内存布局和对齐方式一致。

=== 4.2 混合可变性类型

*问题：* 同时包含可变和不可变组件的类型会导致错误。

*示例：*
```julia
# ❌ 问题类型
struct MixedType
    scalar::Float64        # 不可变
    vector::Vector{Float64}  # 可变
end

t = MixedType(1.0, [2.0, 3.0])
# 使用 Duplicated 可能出错
```

*错误：* `Mixed activity found`

*解决方案：*
添加一层间接性（使用 `Ref`）：

```julia
# ✓ 使用 Ref 包装标量
struct FixedType
    scalar::Ref{Float64}     # 现在可变
    vector::Vector{Float64}
end

t = FixedType(Ref(1.0), [2.0, 3.0])
dt = FixedType(Ref(0.0), zeros(2))
autodiff(Reverse, f, Duplicated(t, dt))  # 现在可以工作
```

=== 4.3 支持的浮点类型

*可微类型：*
- `Float64` ✓
- `Float32` ✓
- `Float16` ✓
- `BFloat16` ✓

*不可微类型：*
- `Int`, `Int32`, `Int64` ✗
- `String` ✗
- `Bool` ✗（但可作为条件）
- `Val{...}` ✗

== 5. 常见错误诊断

=== 5.1 错误类型对照表

#table(
  columns: (auto, auto, auto),
  align: left,
  table.header[*错误*][*常见原因*][*解决方案*],

  [`梯度为零或错误`],
  [临时数组标记为 `Const`],
  [改用 `Duplicated` 并提供导数缓冲],

  [`EnzymeRuntimeActivityError`],
  [变量条件性活跃],
  [启用 `runtimeActivity!()` 或重写代码],

  [`EnzymeMutabilityException`],
  [尝试修改常量或不支持的操作],
  [检查 `reshape`, 复制等操作],

  [`稀疏数组导数为空`],
  [显式零被丢弃],
  [使用 `make_zero()` 初始化],

  [`NaN 或 Inf 梯度`],
  [中间值无穷大未被抑制],
  [启用 `set_strong_zero(true)`],

  [`类型不匹配错误`],
  [`Duplicated` 类型不一致],
  [确保原始值和导数类型相同],
)

=== 5.2 调试步骤

1. *检查函数纯度：*
   - 函数是否有副作用？
   - 是否修改了全局状态？

2. *验证活性注解：*
   - 所有活跃参数用 `Active` 标注？
   - 原地修改的数组用 `Duplicated`？

3. *简化测试：*
   - 创建最小可复现示例
   - 逐步添加复杂性

4. *使用有限差分验证：*
```julia
# Enzyme 梯度
grad_enzyme = autodiff(Reverse, f, Active, Active(x))[1]

# 有限差分验证
eps = 1e-6
grad_fd = (f(x + eps) - f(x - eps)) / (2*eps)

# 比较
@assert abs(grad_enzyme - grad_fd) < 1e-4
```

== 6. 最佳实践

=== 6.1 函数设计

*推荐模式：* 使用可变输出而非返回值

```julia
# 不推荐：分配输出
function compute(x::Vector)
    result = similar(x)
    result .= x.^2
    return result
end

# 推荐：原地修改
function compute!(result, x::Vector)
    result .= x.^2
    return nothing
end

# 使用
result = zeros(10)
dresult = zeros(10)
autodiff(Reverse, compute!,
         Duplicated(result, dresult),
         Duplicated(x, dx))
```

*原因：* 避免内存分配，提高性能。

=== 6.2 初始化策略

```julia
# 为所有可变参数创建导数影子
params = MyParams(W1, b1, W2, b2)
dparams = Enzyme.make_zero(params)  # 自动创建正确结构

# 或手动创建
dparams = MyParams(
    zero(params.W1),
    zero(params.b1),
    zero(params.W2),
    zero(params.b2)
)
```

=== 6.3 性能优化

1. *避免不必要的分配：*
```julia
# 不好：每次调用都分配
function bad(x)
    temp = zeros(100)  # 分配
    # ...
end

# 好：预分配
temp = zeros(100)
function good(x, temp)
    fill!(temp, 0.0)  # 重用
    # ...
end
```

2. *使用类型稳定的代码：*
```julia
# 不好：类型不稳定
function unstable(x)
    if x > 0
        return x  # Float64
    else
        return 0  # Int
    end
end

# 好：类型稳定
function stable(x)
    if x > 0
        return x
    else
        return 0.0  # 明确 Float64
    end
end
```

3. *利用编译器优化：*
```julia
# 使用 @inline 提示
@inline function small_function(x)
    return x^2 + 2*x + 1
end
```

== 7. 实际案例分析

=== 7.1 Richards 方程求解器

*问题代码：*
```julia
function solve_richards_picard!(h_new, h_old, params, ...)
    for iter in 1:max_iter
        # 原地修改
        h_new[i] = h_old[i] + dt * dq / C
    end
    return h_new
end
```

*Enzyme 失败原因：*
- 复杂的原地修改循环
- 条件性收敛检查
- 临时数组分配

*解决方案：*
1. 使用有限差分作为后备
2. 将求解器视为黑盒（如配合 Zygote 使用）
3. 简化求解器逻辑

=== 7.2 神经网络混合模型

*推荐架构：*
```julia
struct HybridModel
    physics_solver  # 可能不可微
    neural_network  # 需要微分
end

function forward(model, x, params)
    # 物理部分：不微分（使用其他方法）
    physics_out = solve_physics(x, params.physics)

    # 神经网络：使用 Enzyme 微分
    nn_out = neural_net(physics_out, params.nn)

    return combine(physics_out, nn_out)
end
```

*策略：* 分离可微和不可微部分。

== 8. 与其他 AD 工具比较

#table(
  columns: (auto, auto, auto, auto),
  align: center,
  table.header[*特性*][*Enzyme*][*Zygote*][*有限差分*],

  [速度], [⚡⚡⚡⚡], [⚡⚡⚡], [🐌],
  [原地修改], [✓ (需标注)], [✗], [✓],
  [控制流], [✓ (需注意)], [✓], [✓],
  [易用性], [⚠️ 中], [✓ 高], [✓✓ 很高],
  [稀疏数组], [⚠️ 需特殊处理], [✓], [✓],
  [CUDA], [✓], [✓], [有限],
  [循环], [✓✓], [✓], [✓],
)

== 9. 配置建议

=== 9.1 项目启动配置

```julia
using Enzyme

# 基本配置
Enzyme.API.strictAliasing!(false)
Enzyme.API.maxtypeoffset!(64)

# 可选：根据需要启用
# Enzyme.API.set_strong_zero(true)      # 处理 Inf/NaN
# Enzyme.API.runtimeActivity!(true)     # 运行时活性
# Enzyme.API.printunnecessary!(true)    # 调试信息
```

=== 9.2 测试策略

```julia
# 始终验证梯度
function test_gradient(f, x; atol=1e-4)
    # Enzyme 梯度
    grad_enzyme = autodiff(Reverse, f, Active, Active(x))[1]

    # 有限差分验证
    eps = 1e-6
    grad_fd = (f(x + eps) - f(x - eps)) / (2*eps)

    @test abs(grad_enzyme - grad_fd) < atol
end
```

== 10. 总结与建议

=== 主要注意事项

1. *原地修改必须用 `Duplicated` 标注*
2. *注意 `min`, `max`, `clamp` 在边界情况的行为*
3. *控制流可能导致运行时活性错误*
4. *避免在热路径中使用 `reshape`*
5. *确保类型稳定和类型匹配*

=== 何时使用 Enzyme

*✓ 适合使用：*
- 需要极致性能的场景
- 控制流简单的函数
- 大量数值计算
- 固定结构的数组操作

*✗ 不适合：*
- 复杂的原地修改逻辑
- 大量动态分配
- 稀疏/特殊数据结构
- 快速原型开发

=== 替代方案

如果 Enzyme 遇到困难，考虑：

1. *Zygote.jl*：更友好，支持更广泛的 Julia 特性
2. *ForwardDiff.jl*：前向模式 AD，适合低维问题
3. *有限差分*：最可靠的后备方案
4. *混合方法*：物理部分用其他方法，神经网络用 Enzyme

#align(center)[
  #box(
    fill: rgb("#e8f4f8"),
    inset: 1em,
    radius: 0.5em,
    [
      *参考资源*

      官方文档：https://enzymead.github.io/Enzyme.jl/

      FAQ：https://enzymead.github.io/Enzyme.jl/dev/faq/

      GitHub：https://github.com/EnzymeAD/Enzyme.jl
    ]
  )
]
