"""数据与预处理诊断脚本。

本脚本只做问题排查，不训练正式模型，不生成 submission。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_DIR = PROJECT_ROOT / "outputs" / "preprocessed"

TARGET_COLUMN = "risk_label"
SESSION_ID_COLUMN = "session_id"
DATE_COLUMN = "op_date"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2

EXPECTED_RAW_TRAIN_COLUMNS = 19
EXPECTED_RAW_TEST_COLUMNS = 18

HIGH_VALUE_FIELDS = [
    "user_name",
    "action",
    "auth_type",
    "ip",
    "ip_location_type_keyword",
    "ip_risk_level",
    "location",
    "device_model",
    "os_type",
    "os_version",
    "browser_type",
    "browser_version",
    "bus_system_code",
    "op_target",
]


def print_section(title: str) -> None:
    """打印诊断章节标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def find_data_file(candidates: list[str], data_dir: Path = DATA_DIR) -> Path:
    """按候选文件名顺序查找数据文件。"""
    for filename in candidates:
        file_path = data_dir / filename
        if file_path.exists():
            return file_path
    raise FileNotFoundError(f"未找到候选数据文件: {candidates}")


def read_csv_auto_sep(file_path: Path) -> tuple[pd.DataFrame, str]:
    """优先按 Tab 读取，若列数仍为 1，再尝试逗号分隔。"""
    df = pd.read_csv(file_path, sep="\t")
    used_sep = "Tab"

    if df.shape[1] == 1:
        comma_df = pd.read_csv(file_path, sep=",")
        if comma_df.shape[1] > df.shape[1]:
            df = comma_df
            used_sep = "Comma"

    df.columns = df.columns.astype(str).str.strip()
    return df, used_sep


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, str, str]:
    """读取原始训练集和测试集。"""
    train_path = find_data_file(["train_dataset(1).csv", "train_dataset.csv"])
    test_path = find_data_file(["test_dataset(1).csv", "test_dataset.csv"])

    train, train_sep = read_csv_auto_sep(train_path)
    test, test_sep = read_csv_auto_sep(test_path)

    return train, test, train_path, test_path, train_sep, test_sep


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """读取预处理后的特征、标签和测试集 session_id。"""
    X_train = pd.read_csv(PREPROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PREPROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PREPROCESSED_DIR / "y_train.csv").iloc[:, 0]
    test_session_id = pd.read_csv(PREPROCESSED_DIR / "test_session_id.csv").iloc[:, 0]
    return X_train, X_test, y_train, test_session_id


def label_distribution(y: pd.Series) -> pd.DataFrame:
    """统计标签数量和比例。"""
    counts = y.value_counts(dropna=False).sort_index()
    ratios = y.value_counts(dropna=False, normalize=True).sort_index()
    return pd.DataFrame({"count": counts, "ratio": ratios})


def diagnose_raw_reading(train: pd.DataFrame, test: pd.DataFrame, train_path: Path, test_path: Path, train_sep: str, test_sep: str) -> list[str]:
    """检查原始数据读取是否正确。"""
    issues = []

    print_section("1. 原始数据读取检查")
    print(f"训练集文件: {train_path.name}, 分隔符: {train_sep}, shape: {train.shape}")
    print(f"测试集文件: {test_path.name}, 分隔符: {test_sep}, shape: {test.shape}")
    print(f"训练集字段: {list(train.columns)}")

    if train.shape[1] != EXPECTED_RAW_TRAIN_COLUMNS:
        issues.append(f"训练集列数异常，期望 {EXPECTED_RAW_TRAIN_COLUMNS} 列，实际 {train.shape[1]} 列。")
    if test.shape[1] != EXPECTED_RAW_TEST_COLUMNS:
        issues.append(f"测试集列数异常，期望 {EXPECTED_RAW_TEST_COLUMNS} 列，实际 {test.shape[1]} 列。")
    if TARGET_COLUMN not in train.columns:
        issues.append("训练集中缺少 risk_label。")

    print(f"risk_label 是否存在于训练集: {TARGET_COLUMN in train.columns}")
    return issues


def diagnose_label(train: pd.DataFrame) -> list[str]:
    """检查 risk_label 取值分布。"""
    issues = []
    print_section("2. risk_label 分布检查")

    if TARGET_COLUMN not in train.columns:
        issues.append("无法检查标签分布，因为训练集缺少 risk_label。")
        return issues

    y = train[TARGET_COLUMN]
    unique_values = sorted(y.dropna().unique().tolist())
    print(f"risk_label 非空唯一值: {unique_values}")
    print(label_distribution(y).to_string())

    if set(unique_values) != {0, 1}:
        issues.append(f"risk_label 不是标准 0/1 二分类标签，当前取值为 {unique_values}。")

    return issues


def diagnose_y_consistency(train: pd.DataFrame, y_train: pd.Series) -> list[str]:
    """检查预处理后的 y_train 是否与原始 risk_label 完全一致。"""
    issues = []
    print_section("3. y_train 与原始 risk_label 一致性检查")

    raw_y = train[TARGET_COLUMN].reset_index(drop=True)
    processed_y = y_train.reset_index(drop=True)

    same_length = len(raw_y) == len(processed_y)
    same_values = same_length and raw_y.equals(processed_y)
    mismatch_count = 0 if same_values else int((raw_y != processed_y).sum()) if same_length else None

    print(f"原始 risk_label 长度: {len(raw_y)}")
    print(f"预处理 y_train 长度: {len(processed_y)}")
    print(f"是否完全一致: {same_values}")
    print(f"不一致数量: {mismatch_count}")

    if not same_values:
        issues.append("y_train.csv 与原始 risk_label 不完全一致。")

    return issues


def diagnose_split(y_train: pd.Series) -> list[str]:
    """检查 train_test_split 后训练集和验证集标签分布。"""
    issues = []
    print_section("4. train_test_split 标签分布检查")

    _X_dummy = np.zeros((len(y_train), 1))
    _, _, y_tr, y_val = train_test_split(
        _X_dummy,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    full_dist = label_distribution(y_train)
    train_dist = label_distribution(pd.Series(y_tr))
    val_dist = label_distribution(pd.Series(y_val))

    print("整体标签分布:")
    print(full_dist.to_string())
    print("\n划分后训练集标签分布:")
    print(train_dist.to_string())
    print("\n划分后验证集标签分布:")
    print(val_dist.to_string())

    class_count_ok = pd.Series(y_tr).nunique() == 2 and pd.Series(y_val).nunique() == 2
    if not class_count_ok:
        issues.append("划分后的训练集或验证集缺少某个标签类别。")

    return issues


def diagnose_feature_matrix(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
    """检查预处理后的 X_train 是否存在明显异常。"""
    issues = []
    print_section("5. X_train 特征矩阵异常检查")

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_train/X_test 字段完全一致: {list(X_train.columns) == list(X_test.columns)}")

    missing_total = int(X_train.isna().sum().sum())
    all_zero_columns = X_train.columns[(X_train == 0).all(axis=0)].tolist()
    constant_columns = X_train.columns[X_train.nunique(dropna=False) <= 1].tolist()
    negative_one_ratios = {
        col: float((X_train[col] == -1).mean())
        for col in X_train.select_dtypes(include=[np.number]).columns
        if (X_train[col] == -1).any()
    }
    missing_encoded_ratios = {
        col: float(X_train[col].mean())
        for col in X_train.columns
        if "__MISSING__" in col and pd.api.types.is_numeric_dtype(X_train[col])
    }

    print(f"缺失值总数: {missing_total}")
    print(f"全零列数量: {len(all_zero_columns)}")
    print(f"常数列数量: {len(constant_columns)}")
    print(f"全零列示例: {all_zero_columns[:20]}")
    print(f"常数列示例: {constant_columns[:20]}")
    print(f"包含 -1 填充值的列及比例: {negative_one_ratios}")
    print(f"缺失值哑变量列及占比: {missing_encoded_ratios}")

    if missing_total > 0:
        issues.append("X_train 中仍存在缺失值。")
    if constant_columns:
        issues.append(f"X_train 中存在 {len(constant_columns)} 个常数列。")
    if any(ratio > 0.8 for ratio in missing_encoded_ratios.values()):
        issues.append("存在缺失占比超过 80% 的缺失值编码列。")

    return issues


def column_matches_feature(
    columns: list[str],
    raw_field: str,
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
) -> list[str]:
    """查找某个原始字段在预处理结果中的保留列。"""
    if raw_field in columns:
        return [raw_field]

    # 结合原始取值生成 One-Hot 后的精确列名，避免 ip 误匹配 ip_location 等字段。
    if train is not None and test is not None and raw_field in train.columns and raw_field in test.columns:
        values = (
            pd.concat([train[raw_field], test[raw_field]], axis=0)
            .fillna("__MISSING__")
            .astype(str)
            .unique()
            .tolist()
        )
        expected_columns = {f"{raw_field}_{value}" for value in values}
        return [col for col in columns if col in expected_columns]

    return [col for col in columns if col.startswith(f"{raw_field}_")]


def diagnose_one_hot_and_field_retention(train: pd.DataFrame, test: pd.DataFrame, X_train: pd.DataFrame) -> list[str]:
    """检查 One-Hot 编码后字段是否保留。"""
    issues = []
    print_section("6. One-Hot 编码字段保留检查")

    columns = list(X_train.columns)
    time_columns = ["op_year", "op_month", "op_day", "op_hour", "op_weekday"]
    raw_feature_fields = [
        col
        for col in train.columns
        if col not in {TARGET_COLUMN, SESSION_ID_COLUMN, DATE_COLUMN}
    ]

    rows = []
    for field in raw_feature_fields:
        combined_nunique = pd.concat([train[field], test[field]], axis=0).nunique(dropna=False)
        matches = column_matches_feature(columns, field, train, test)
        rows.append(
            {
                "raw_field": field,
                "combined_nunique": int(combined_nunique),
                "encoded_or_kept_cols": len(matches),
                "status": "保留" if matches else "未保留",
                "note": "常数列被预处理删除" if combined_nunique <= 1 and not matches else "",
            }
        )

    retention_df = pd.DataFrame(rows)
    print(retention_df.to_string(index=False))
    print(f"\n时间派生字段是否存在: {all(col in X_train.columns for col in time_columns)}")
    print(f"时间派生字段: {[col for col in time_columns if col in X_train.columns]}")

    lost_important_fields = []
    for field in HIGH_VALUE_FIELDS:
        if field in train.columns:
            matches = column_matches_feature(columns, field, train, test)
            combined_nunique = pd.concat([train[field], test[field]], axis=0).nunique(dropna=False)
            if not matches and combined_nunique > 1:
                lost_important_fields.append(field)

    if lost_important_fields:
        issues.append(f"高价值字段疑似被错误删除: {lost_important_fields}")

    return issues


def diagnose_op_date(train: pd.DataFrame, test: pd.DataFrame, X_train: pd.DataFrame) -> list[str]:
    """检查 op_date 是否正确解析并生成时间特征。"""
    issues = []
    print_section("7. op_date 解析检查")

    train_dt = pd.to_datetime(train[DATE_COLUMN], errors="coerce")
    test_dt = pd.to_datetime(test[DATE_COLUMN], errors="coerce")
    time_columns = ["op_year", "op_month", "op_day", "op_hour", "op_weekday"]

    print(f"训练集 op_date 无法解析数量: {int(train_dt.isna().sum())}")
    print(f"测试集 op_date 无法解析数量: {int(test_dt.isna().sum())}")
    print(f"训练集 op_date 范围: {train_dt.min()} ~ {train_dt.max()}")
    print(f"测试集 op_date 范围: {test_dt.min()} ~ {test_dt.max()}")
    print(f"训练集小时分布: {train_dt.dt.hour.value_counts().sort_index().to_dict()}")
    print({col: col in X_train.columns for col in time_columns})

    if train_dt.isna().any() or test_dt.isna().any():
        issues.append("op_date 存在无法解析的值。")
    if not all(col in X_train.columns for col in time_columns):
        issues.append("op_date 的时间派生字段未完整保留。")

    return issues


def diagnose_high_value_fields(train: pd.DataFrame, test: pd.DataFrame, X_train: pd.DataFrame) -> list[str]:
    """检查用户、IP、设备、系统等字段是否被错误删除。"""
    issues = []
    print_section("8. 高价值字段保留检查")

    rows = []
    for field in HIGH_VALUE_FIELDS:
        if field not in train.columns:
            rows.append(
                {
                    "field": field,
                    "train_nunique": None,
                    "test_nunique": None,
                    "preprocessed_cols": 0,
                    "status": "原始数据缺失",
                }
            )
            continue

        matches = column_matches_feature(list(X_train.columns), field, train, test)
        train_nunique = train[field].nunique(dropna=False)
        test_nunique = test[field].nunique(dropna=False)
        combined_nunique = pd.concat([train[field], test[field]], axis=0).nunique(dropna=False)

        if matches:
            status = "已保留"
        elif combined_nunique <= 1:
            status = "常数列删除"
        else:
            status = "疑似误删"
            issues.append(f"字段 {field} 非常数但未在预处理特征中找到。")

        rows.append(
            {
                "field": field,
                "train_nunique": int(train_nunique),
                "test_nunique": int(test_nunique),
                "preprocessed_cols": len(matches),
                "status": status,
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))
    return issues


def field_label_relation(train: pd.DataFrame, field: str) -> tuple[pd.DataFrame | None, dict[str, float | int | str | None]]:
    """计算单个原始字段与 risk_label 的简单关系。"""
    temp = train[[field, TARGET_COLUMN]].copy()
    temp[field] = temp[field].fillna("__MISSING__").astype(str)
    nunique = temp[field].nunique(dropna=False)

    summary = {
        "field": field,
        "nunique": int(nunique),
        "min_mean": None,
        "max_mean": None,
        "mean_gap": None,
        "note": "",
    }

    if field == SESSION_ID_COLUMN:
        summary["note"] = "session_id 基本是逐行唯一，不适合按类别聚合解释。"
        return None, summary
    if field == DATE_COLUMN:
        parsed = pd.to_datetime(train[field], errors="coerce")
        temp["op_hour"] = parsed.dt.hour.astype("Int64").astype(str)
        grouped = (
            temp.groupby("op_hour")[TARGET_COLUMN]
            .agg(["count", "mean"])
            .sort_values("mean", ascending=False)
        )
        summary["field"] = "op_date -> op_hour"
        summary["nunique"] = int(grouped.shape[0])
        summary["min_mean"] = float(grouped["mean"].min())
        summary["max_mean"] = float(grouped["mean"].max())
        summary["mean_gap"] = float(grouped["mean"].max() - grouped["mean"].min())
        summary["note"] = "原始 op_date 唯一值过多，此处按小时聚合。"
        return grouped, summary

    grouped = (
        temp.groupby(field)[TARGET_COLUMN]
        .agg(["count", "mean"])
        .sort_values(["mean", "count"], ascending=[False, False])
    )
    summary["min_mean"] = float(grouped["mean"].min())
    summary["max_mean"] = float(grouped["mean"].max())
    summary["mean_gap"] = float(grouped["mean"].max() - grouped["mean"].min())

    return grouped, summary


def diagnose_field_label_relations(train: pd.DataFrame) -> list[dict[str, float | int | str | None]]:
    """输出每个原始字段与 risk_label 的简单关系。"""
    print_section("9. 原始字段与 risk_label 的简单关系")

    relation_summaries = []
    fields = [col for col in train.columns if col != TARGET_COLUMN]

    for field in fields:
        grouped, summary = field_label_relation(train, field)
        relation_summaries.append(summary)

        print(f"\n字段: {summary['field']}")
        print(f"唯一值数量: {summary['nunique']}")
        if summary["note"]:
            print(f"说明: {summary['note']}")
        if grouped is not None:
            print(grouped.head(12).to_string())
            print(
                "risk_label 均值范围: "
                f"{summary['min_mean']:.6f} ~ {summary['max_mean']:.6f}, "
                f"差值: {summary['mean_gap']:.6f}"
            )

    print("\n字段关系差值汇总，按 risk_label 均值差值降序:")
    relation_df = pd.DataFrame(relation_summaries)
    relation_df = relation_df.sort_values("mean_gap", ascending=False, na_position="last")
    print(relation_df.to_string(index=False))

    return relation_summaries


def print_final_conclusion(issues: list[str], relation_summaries: list[dict[str, float | int | str | None]]) -> None:
    """根据诊断结果输出总体结论。"""
    print_section("10. 诊断结论")

    if issues:
        print("发现的问题:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("未发现数据读取、标签对齐、分层切分或高价值字段保留方面的硬性错误。")

    relation_df = pd.DataFrame(relation_summaries)
    valid_relation_df = relation_df.dropna(subset=["mean_gap"]).sort_values("mean_gap", ascending=False)
    weak_signal_fields = valid_relation_df[valid_relation_df["mean_gap"] < 0.05]["field"].tolist()

    print("\n最可能原因初步判断:")
    if issues:
        print("- 优先排查上方硬性问题，它们可能直接导致验证 AUC 接近 0.5。")
    elif len(valid_relation_df) > 0:
        top_rows = valid_relation_df.head(5)
        print("- 预处理主流程没有明显把标签、分层划分或高价值字段弄错。")
        print("- AUC 接近 0.5 更可能来自当前特征与 risk_label 的可分性较弱，或仅有简单 One-Hot/粗粒度时间特征不足以表达有效模式。")
        print("- 单字段关系中差异最大的字段如下:")
        print(top_rows[["field", "nunique", "min_mean", "max_mean", "mean_gap"]].to_string(index=False))
        if len(weak_signal_fields) == len(valid_relation_df):
            print("- 所有单字段的标签均值差异都很小，说明当前原始字段的单变量信号整体偏弱。")
    else:
        print("- 缺少可用于判断字段关系的统计结果。")

    print("\n注意: 本脚本未训练正式模型，未生成 submission。")


def main() -> None:
    """执行完整诊断流程。"""
    issues: list[str] = []

    train, test, train_path, test_path, train_sep, test_sep = load_raw_data()
    X_train, X_test, y_train, _test_session_id = load_preprocessed_data()

    issues.extend(diagnose_raw_reading(train, test, train_path, test_path, train_sep, test_sep))
    issues.extend(diagnose_label(train))
    issues.extend(diagnose_y_consistency(train, y_train))
    issues.extend(diagnose_split(y_train))
    issues.extend(diagnose_feature_matrix(X_train, X_test))
    issues.extend(diagnose_one_hot_and_field_retention(train, test, X_train))
    issues.extend(diagnose_op_date(train, test, X_train))
    issues.extend(diagnose_high_value_fields(train, test, X_train))
    relation_summaries = diagnose_field_label_relations(train)
    print_final_conclusion(issues, relation_summaries)


if __name__ == "__main__":
    main()
