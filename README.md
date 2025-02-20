# 2025 NCAA篮球锦标赛预测模型

这个项目使用机器学习方法预测2025年NCAA篮球锦标赛的比赛结果。该模型基于历史比赛数据，结合多个高级特征和集成学习方法，为每场比赛提供胜率预测。

## 功能特点

- 使用2003-2024年的历史数据进行训练
- 综合多个机器学习模型（LightGBM、XGBoost、神经网络）
- 丰富的特征工程，包括：
  - 球队基础统计数据
  - 历史交手记录
  - 种子排名特征
  - 动量指标
  - 赛季表现趋势
- 使用Brier Score评估预测准确性
- 动态权重集成提高预测稳定性

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/augensternnnnn21/2025-NCAA-Basketball-Tournaments.git
cd 2025-NCAA-Basketball-Tournaments
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：
   - 将NCAA比赛数据文件放在项目根目录
   - 确保包含所有必需的CSV文件（见数据文件列表）

2. 运行模型：
```bash
python src/main.py
```

3. 查看结果：
   - 模型会在`models/`目录下保存训练好的模型
   - 预测结果将包含每场比赛的胜率预测

## 数据文件说明

必需的数据文件：
- `MRegularSeasonCompactResults.csv`: 常规赛比赛结果
- `MNCAATourneyCompactResults.csv`: 锦标赛比赛结果
- `MTeams.csv`: 球队信息
- `MNCAATourneySeeds.csv`: 锦标赛种子排名

## 项目结构

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── main.py
│   ├── data/
│   │   └── data_loader.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── models/
│       └── model_trainer.py
└── models/
    ├── lightgbm_model.pkl
    ├── scaler.pkl
    └── selected_features.pkl
```

## 模型评估

- 使用2003-2024年的历史数据进行训练和验证
- 采用Brier Score作为主要评估指标
- 通过交叉验证确保模型稳定性
- 集成多个模型以提高预测准确性

## 注意事项

- 需要足够的内存来处理大量历史数据
- 首次运行可能需要较长时间来处理数据和训练模型
- 建议使用GPU来加速神经网络训练（如果可用）

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。在提交代码前，请确保：
1. 代码符合PEP 8规范
2. 添加适当的注释和文档
3. 所有测试通过