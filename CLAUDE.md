# 1 基本原则

- linux极简主义风格，脚本与函数要分开

- 尽可能的少用print，代码要精简，代码行数要尽可能的短

- 模型的方案，采用typst编写，保存在docs中

- 新加的方法都要有测试

- 代码风格，请参考<dev/Framework.jl>, <Richards/2_04.jl>

- plan模式，对话内容，请保存在dev文件夹下，采用typst记录

- 成熟的代码放到了src文件夹下。其中不同类型的函数，应该放到不同的文件中，方便管理。

- 如果包有缺失，请安装到project的环境中

- csv数据读取请采用RTableTools, `df = fread(f_csv)`

# 2 Soil

**土壤湿度观测**: 5, 10, 20, 50, 100cm

```R
time,P_CALC,SOIL_MOISTURE_5,SOIL_MOISTURE_10,SOIL_MOISTURE_20,SOIL_MOISTURE_50,SOIL_MOISTURE_100
```

**输入数据**：data/SM_AR_Batesville_8_WNW_2024.csv

**模拟**：输入P，土壤水力参数，采用深度学习框架进行优化，求解SM。


# TODO

~~1. 在<Richards/2_04.jl>的基础上，使用真实数据，检查模型模拟效果。~~
✓ 已完成，见 <Richards/NeuralODE.jl> 和 <Richards/test_neuralode.jl>

~~2. 学习<dev/Framework.jl>的设计模式，编写适用于土壤水运动的train和evaluate函数。~~
✓ 已完成，见 <src/NeuralODE.jl> 和 <test/test_neuralode.jl>
- train函数：采用MSE损失，使用Optimization.jl框架
- evaluate函数：对每层土壤水计算GOF指标（NSE, KGE, R2, RMSE等）
- predict函数：前向预测土壤水分

3. 你的evaluate写的不对，你目前评估的是某一时刻的精度，例如`evaluate(hybrid_node, θ₀, ps, θ_obs)`
θ₀是t=0时的5层土壤含水量，而我需要评估的是全部时刻的表现
