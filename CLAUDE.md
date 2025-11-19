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

3. 模型结构不合理，现在的模型精度非常有限。侧向壤中流 考虑不周，导致深层土壤的缓慢波动 无法捕捉。深层土壤K下降，导致土壤中层形成perched water table，导致侧向壤中流。现在的模型是如何考虑该问题的？

[训练后评估]
5×11 DataFrame
 Row │ layer  depth    NSE          R2         KGE         R         RMSE       MAE        bias        bias_perc  n_valid 
     │ Int64  Float32  Float32      Float32    Float32     Float32   Float32    Float32    Float32     Float32    Int64   
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     1      5.0   -2.80611    0.454625    0.169174   0.674259  0.0377724  0.0340293   0.0340293   19.1346       240
   2 │     2     10.0    0.0904944  0.509846    0.536672   0.714035  0.013808   0.0118379   0.0093105    3.99921      240
   3 │     3     20.0   -0.896418   0.568634    0.64615    0.754078  0.0175008  0.0154263   0.0153822    6.46322      240
   4 │     4     50.0  -30.4522     0.0472344   0.0490863  0.217335  0.0214305  0.0204946  -0.0204946   -7.87106      240
   5 │     5    100.0  -42.7889     0.70419    -0.865722   0.83916   0.0286235  0.0271615  -0.0271603  -10.5118       240
