# ML_CourseDesign

基于机器学习的系统认证风险预测模型设计与实现。

## 项目简介

本项目基于 DataFountain「系统认证风险预测」竞赛数据，完成了数据探索分析（EDA）、数据预处理、特征工程、模型训练及模型评价等完整机器学习流程，并生成符合竞赛要求的提交文件。

## 项目结构

```
ML_CourseDesign
│
├── data
│   ├── train_dataset.csv
│   ├── test_dataset.csv
│   └── submit_example.csv
│
├── notebooks
│   └── 01_EDA.ipynb
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── diagnose.py
│   └── utils.py
│
├── outputs
│   ├── preprocessed
│   ├── model_results.csv
│   ├── submission.csv
│   ├── submission_label.csv
│   ├── submission_proba.csv
│   ├── label_distribution.png
│   └── hour_distribution.png
│
├── requirements.txt
└── README.md
```

## 开发环境

- Python 3.11
- pandas
- numpy
- matplotlib
- scikit-learn

## 运行方式

首先进行数据预处理：

```bash
python src/preprocess.py
```

然后进行模型训练：

```bash
python src/train.py
```

## 模型

本项目实现了以下模型：

- Logistic Regression（逻辑回归）
- Random Forest（随机森林）

模型评价指标采用 **AUC（Area Under Curve）**。

## 输出结果

程序运行后将在 `outputs/` 目录生成：

- model_results.csv
- submission.csv
- submission_label.csv
- submission_proba.csv

## 数据来源

DataFountain 系统认证风险预测竞赛数据，仅用于课程设计学习与研究。
