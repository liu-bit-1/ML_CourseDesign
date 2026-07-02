"""传统机器学习训练脚本。

本脚本只使用已经完成的数据预处理结果进行传统机器学习建模，不包含任何
深度学习、神经网络、BERT 或 Transformer 相关逻辑。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = PROJECT_ROOT / "outputs" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """读取预处理阶段生成的训练特征、测试特征、标签和测试集 session_id。"""
    X_train_path = PREPROCESSED_DIR / "X_train.csv"
    X_test_path = PREPROCESSED_DIR / "X_test.csv"
    y_train_path = PREPROCESSED_DIR / "y_train.csv"
    test_session_id_path = PREPROCESSED_DIR / "test_session_id.csv"

    required_files = [X_train_path, X_test_path, y_train_path, test_session_id_path]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"缺少预处理输出文件，请先运行 src/preprocess.py:\n{missing_text}")

    # 读取已经对齐好的训练集和测试集特征
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # 读取训练标签，默认第一列为 risk_label
    y_train_df = pd.read_csv(y_train_path)
    y_train = y_train_df.iloc[:, 0]

    # 读取测试集 session_id，用于生成 DataFountain 提交文件
    test_session_id_df = pd.read_csv(test_session_id_path)
    test_session_id = test_session_id_df.iloc[:, 0]

    return X_train, X_test, y_train, test_session_id


def get_positive_class_proba(model, X: pd.DataFrame) -> pd.Series:
    """获取预测为 1 的概率，用于计算 AUC 和生成概率版提交文件。"""
    if 1 not in model.classes_:
        raise ValueError("模型类别中未找到正类标签 1，无法计算正类概率。")

    positive_class_index = list(model.classes_).index(1)
    return pd.Series(model.predict_proba(X)[:, positive_class_index], index=X.index)


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, str, pd.DataFrame]:
    """训练 Logistic Regression 和 Random Forest，并使用验证集 AUC 进行比较。"""
    # 使用分层抽样划分训练集和验证集，保持标签比例一致
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # 定义两个传统机器学习模型，不使用任何深度学习模型
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for model_name, model in models.items():
        # 训练当前模型
        model.fit(X_tr, y_tr)
        trained_models[model_name] = model

        # 使用验证集预测正类概率，并计算 AUC
        val_proba = get_positive_class_proba(model, X_val)
        validation_auc = roc_auc_score(y_val, val_proba)

        print(f"{model_name} validation AUC: {validation_auc:.6f}")
        results.append(
            {
                "model": model_name,
                "validation_auc": validation_auc,
            }
        )

    results_df = pd.DataFrame(results).sort_values("validation_auc", ascending=False)
    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    return best_model, best_model_name, results_df


def save_model_results(results_df: pd.DataFrame) -> Path:
    """保存模型验证集 AUC 对比结果。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "model_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    return results_path


def save_submissions(
    model,
    X_test: pd.DataFrame,
    test_session_id: pd.Series,
) -> tuple[Path, Path]:
    """生成正式概率提交文件和硬标签备用文件。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # AUC 评测关注样本排序，正式提交使用预测为 1 的概率作为风险分数
    test_proba = get_positive_class_proba(model, X_test)
    submission = pd.DataFrame(
        {
            "id": test_session_id,
            "ret": test_proba,
        }
    ).sort_values("id", ascending=True)

    # 另存硬分类标签，仅用于本地检查，不作为 AUC 正式提交文件
    test_pred = model.predict(X_test)
    submission_label = pd.DataFrame(
        {
            "id": test_session_id,
            "ret": test_pred.astype(int),
        }
    ).sort_values("id", ascending=True)

    submission_path = OUTPUT_DIR / "submission.csv"
    submission_label_path = OUTPUT_DIR / "submission_label.csv"

    submission.to_csv(submission_path, index=False, encoding="utf-8", float_format="%.6f")
    submission_label.to_csv(submission_label_path, index=False, encoding="utf-8")

    return submission_path, submission_label_path


def main() -> None:
    """运行传统机器学习训练、验证和提交文件生成流程。"""
    X_train, X_test, y_train, test_session_id = load_preprocessed_data()

    best_model, best_model_name, results_df = train_and_evaluate_models(X_train, y_train)
    results_path = save_model_results(results_df)
    submission_path, submission_label_path = save_submissions(best_model, X_test, test_session_id)

    print(f"Best model: {best_model_name}")
    print(f"Model results saved to: {results_path}")
    print(f"Submission saved to: {submission_path}")
    print(f"Label submission saved to: {submission_label_path}")


if __name__ == "__main__":
    main()
