# Soil 结构体说明

## 定义

使用 `@with_kw` 宏定义的 `Soil` 结构体，用于管理 Richards 方程求解中的所有土壤状态变量。

```julia
@with_kw mutable struct Soil{T<:AbstractFloat}
  K::Vector{T}           # 节点水力传导度 [cm/s]
  K_interface::Vector{T} # 界面水力传导度 [cm/s]
  ψ::Vector{T}           # 土壤水势 [cm]
  q::Vector{T}           # 达西通量 [cm/s]
  θ_prev::Vector{T}      # 上一时刻含水量 [-]
  θ::Vector{T}           # 当前含水量 [-]
end
```

## 字段说明

| 字段 | 类型 | 单位 | 维度 | 说明 |
|------|------|------|------|------|
| `K` | `Vector{T}` | cm/s | n | 各层的水力传导度（节点值，van Genuchten模型） |
| `K_interface` | `Vector{T}` | cm/s | n+1 | 各界面的水力传导度（通过算术/几何平均计算） |
| `ψ` | `Vector{T}` | cm | n | 各层的土壤水势（压力水头） |
| `q` | `Vector{T}` | cm/s | n+1 | 各界面的达西通量 |
| `θ_prev` | `Vector{T}` | - | n | 上一时刻的含水量 |
| `θ` | `Vector{T}` | - | n | 当前时刻的含水量 |

### 界面与节点的区别

```
层 (节点)          界面
    1         ←  界面 1 (顶部边界)
  [K[1]]
    ↓         ←  界面 2 (内部) K_interface[2] = (K[1]+K[2])/2
    2
  [K[2]]
    ↓         ←  界面 3 (内部) K_interface[3] = (K[2]+K[3])/2
    3
  [K[3]]
    ↓         ←  界面 4 (内部) K_interface[4] = (K[3]+K[4])/4
    4
  [K[4]]
    ↓         ←  界面 5 (内部) K_interface[5] = (K[4]+K[5])/2
    5
  [K[5]]
              ←  界面 6 (底部边界)
```

- **节点值 (n个)**：定义在每一层的中心
- **界面值 (n+1个)**：定义在层与层之间的界面

### 为什么需要 K_interface？

在有限差分/有限体积方法中，通量计算发生在界面上：

**达西定律在界面 i 的应用：**
```julia
q[i] = -K_interface[i] * (∂ψ/∂z + 1)
```

**界面传导度的计算方法：**
1. **算术平均**（默认）：
   ```julia
   K_interface[i] = (K[i-1] + K[i]) / 2
   ```
   - 简单易用
   - 适用于水力性质变化不大的情况

2. **几何平均**（更物理）：
   ```julia
   K_interface[i] = sqrt(K[i-1] * K[i])
   ```
   - 更符合非均质介质的物理特性
   - 避免高导水率"短路"效应

3. **调和平均**（保守）：
   ```julia
   K_interface[i] = 2 / (1/K[i-1] + 1/K[i])
   ```
   - 最保守
   - 适用于存在低导水率层的情况

将 `K_interface` 存储在结构体中的优势：
- ✓ 便于调试和可视化
- ✓ 可以尝试不同的平均方法
- ✓ 减少重复计算
- ✓ 完整记录模型状态

## 使用方法

### 1. 创建实例

```julia
# 方法1：指定层数（默认 Float32）
soil = Soil(5)  # 5层土壤

# 方法2：指定层数和类型
soil = Soil(10, Float64)  # 10层，使用 Float64

# 方法3：使用关键字参数（@with_kw 的优势）
soil = Soil(
  K = zeros(Float32, 5),
  K_interface = zeros(Float32, 6),
  ψ = zeros(Float32, 5),
  q = zeros(Float32, 6),
  θ_prev = fill(0.25f0, 5),
  θ = fill(0.25f0, 5)
)
```

### 2. 初始化状态

```julia
soil = Soil(5, Float32)

# 设置初始含水量
soil.θ .= 0.25f0
soil.θ_prev .= soil.θ

# 其他变量会在求解过程中自动计算
```

### 3. 使用改进的 Richards 求解器

```julia
# 土壤参数
soil_params = (
  θ_s = 0.45f0,  # 饱和含水量
  θ_r = 0.05f0,  # 残余含水量
  Ks = 0.01f0,   # 饱和导水率
  α = 0.02f0,    # van Genuchten 参数
  n = 1.5f0      # van Genuchten 参数
)

# 时间和空间步长
Δt = 3600.0f0  # 1小时
Δz = 10.0f0    # 10 cm

# 运行求解器（原地修改）
richards_explicit_step!(soil, soil_params, Δt, Δz)

# 访问结果
println("含水量: ", soil.θ)
println("节点水力传导度: ", soil.K)
println("界面水力传导度: ", soil.K_interface)
println("通量: ", soil.q)
```

## 结构体优势

### 1. **状态变量集中管理**
所有相关变量存储在一个对象中，避免了多个独立数组的管理。

```julia
# 之前：需要管理多个变量
K = zeros(n)
K_interface = zeros(n+1)  # 容易忘记！
ψ = zeros(n)
q = zeros(n+1)
θ_prev = zeros(n)
θ = zeros(n)
# 6 个独立变量，容易出错

# 现在：一个结构体搞定
soil = Soil(n)
# 所有变量自动初始化，维度正确
```

### 2. **类型安全**
通过泛型参数 `T<:AbstractFloat` 确保类型一致性。

```julia
soil = Soil(5, Float32)  # 所有字段都是 Float32
# 不会意外混合 Float32 和 Float64
```

### 3. **关键字参数初始化**
`@with_kw` 宏提供了灵活的初始化方式。

```julia
# 可以只设置部分字段
soil = Soil(
  K = ones(Float32, 5),
  θ = fill(0.3f0, 5),
  # 其他字段使用默认值
)
```

### 4. **便于保存和传递状态**
结构体可以方便地保存中间状态或传递给其他函数。

```julia
# 保存状态历史
history = Soil[]
for t in 1:100
  richards_explicit_step!(soil, params, Δt, Δz)
  push!(history, deepcopy(soil))  # 保存当前状态
end
```

### 5. **可扩展性**
可以轻松添加新字段而不影响现有代码。

```julia
@with_kw mutable struct SoilExtended{T<:AbstractFloat}
  # 原有字段
  K::Vector{T}
  ψ::Vector{T}
  q::Vector{T}
  θ_prev::Vector{T}
  θ::Vector{T}

  # 新增字段
  temperature::Vector{T} = zeros(T, length(θ))
  solute_conc::Vector{T} = zeros(T, length(θ))
end
```

## 与原版本对比

### 原版本（分散的数组）

```julia
function richards_explicit_step(θ_old, soil_params, Δt, Δz)
  n_layers = length(θ_old)
  θ_new = copy(θ_old)

  # 需要在函数内部临时计算 K, ψ, q
  # 这些中间变量无法访问

  for i in 2:n_layers-1
    K_upper = ...  # 重复计算
    # ...
  end

  return θ_new  # 只返回含水量
end
```

**缺点：**
- 中间变量（K, K_interface, ψ, q）无法访问
- 每次调用都重新计算
- 无法查看通量、界面传导度等诊断信息
- K_interface 每次都在循环内部重新计算，无法复用

### 新版本（结构体）

```julia
function richards_explicit_step!(soil::Soil, soil_params, Δt, Δz)
  # 保存历史
  soil.θ_prev .= soil.θ

  # 计算并存储水力性质
  for i in 1:n_layers
    soil.K[i] = hydraulic_conductivity(...)
    soil.ψ[i] = soil_water_potential(...)
  end

  # 计算并存储通量
  for i in 2:n_layers
    soil.q[i] = ...
  end

  # 更新含水量
  soil.θ[i] = ...

  return nothing  # 原地修改
end
```

**优点：**
- 所有中间变量可访问（K, ψ, q）
- 便于调试和诊断
- 支持原地修改，减少内存分配
- 状态完全可见和可控

## 使用示例

完整的使用示例可参考代码中的 `demo_soil_struct()` 函数：

```julia
# 运行演示
demo_soil_struct()
```

输出示例：
```
============================================================
演示 Soil 结构体使用
============================================================

初始状态:
  含水量 θ: [0.25, 0.25, 0.25, 0.25, 0.25]
  水力传导度 K: [0.0, 0.0, 0.0, 0.0, 0.0]
  土壤水势 ψ: [0.0, 0.0, 0.0, 0.0, 0.0]
  通量 q: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

运行一步后:
  含水量 θ: [0.25, 0.2501, 0.2502, 0.2503, 0.25]
  水力传导度 K: [0.0031, 0.0031, 0.0031, 0.0031, 0.0031]
  土壤水势 ψ: [-45.2, -45.2, -45.2, -45.2, -45.2]
  通量 q: [0.0, 0.0031, 0.0031, 0.0031, 0.0]
...
```

## 与 Enzyme.jl 的兼容性

使用结构体时需要注意 Enzyme 的要求：

### ✓ 兼容的用法

```julia
# 使用 Duplicated 标注
soil = Soil(5)
dsoil = Soil(5)  # 梯度容器

Enzyme.autodiff(
  Reverse,
  my_function,
  Duplicated(soil, dsoil),
  ...
)
```

### ⚠️ 注意事项

1. **类型稳定性**：确保所有字段类型在编译时确定
2. **可变性**：使用 `mutable struct` 支持原地修改
3. **避免混合活性**：所有字段应该都是可微的或都是常量

## 总结

`Soil` 结构体提供了一个清晰、类型安全且易于扩展的方式来管理 Richards 方程求解中的所有状态变量。通过 `@with_kw` 宏，它还提供了灵活的初始化选项。这种设计模式特别适合复杂的物理模型，其中多个相关变量需要协同管理。

**推荐使用场景：**
- 需要访问中间计算结果（K, ψ, q）
- 需要保存状态历史
- 需要与其他物理过程耦合（如热传输、溶质运移）
- 需要类型安全和可维护性

**何时使用原版本：**
- 只关心最终含水量
- 简单的一次性计算
- 不需要诊断信息
