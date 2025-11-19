# 1 基本原则

- linux极简主义风格

- 模型的方案，采用typst编写，保存在docs中

- 新加的方法都要有测试

- 代码风格，请参考<Richards/2_04.jl>, <Richards/Soil.jl>


# 2 Soil

**土壤湿度观测**: 5, 10, 20, 50, 100cm

```R
time,P_CALC,SOIL_MOISTURE_5,SOIL_MOISTURE_10,SOIL_MOISTURE_20,SOIL_MOISTURE_50,SOIL_MOISTURE_100
```

**输入数据**：data/SM_AR_Batesville_8_WNW_2024.csv

**模拟**：输入P，土壤水力参数，采用深度学习框架进行优化，求解SM。
