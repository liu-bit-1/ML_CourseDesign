# Machine Learning Course Design

这是一个机器学习课程设计项目的初始化结构。当前阶段仅完成项目目录、依赖文件和基础代码框架准备，尚未开始训练模型或填写实验结果。

## 项目结构

```text
ML_CourseDesign/
├── data/
│   ├── train_dataset.csv
│   ├── test_dataset.csv
│   └── submit_example.csv
├── notebooks/
│   └── 01_EDA.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── README.md
└── requirements.txt
```

## 环境要求

- Python 3.11
- 依赖库见 `requirements.txt`

## 安装依赖

建议先创建并激活虚拟环境，然后安装项目依赖：

```bash
pip install -r requirements.txt
```

## 运行方式

1. 数据文件放置在 `data/` 目录中。
2. 使用 `notebooks/01_EDA.ipynb` 进行初步数据探索。
3. 在 `src/preprocess.py` 中整理后续数据预处理流程。
4. 在 `src/train.py` 中整理后续模型训练入口。
5. 在 `src/utils.py` 中放置通用辅助函数。

## 当前状态

- 已建立基础项目结构。
- 已准备 EDA notebook 模板。
- 已准备预处理、训练和工具函数脚本框架。
- 暂未进行模型训练。
- 暂未填写实验结果。
