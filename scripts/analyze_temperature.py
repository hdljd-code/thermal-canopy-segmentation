import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
TEST_SPLIT = DATASET_DIR / "splits/test.txt"
TEMPERATURE_DIR = DATASET_DIR / "temperature"
MASK_DIR = DATASET_DIR / "masks"
BASELINE_PRED_DIR = PROJECT_ROOT / "predictions/baseline"
PROPOSED_PRED_DIR = PROJECT_ROOT / "predictions/proposed"
OUTPUT_DIR = PROJECT_ROOT / "outputs/temperature_analysis"

EXPECTED_IMAGE_COUNT = 115
PER_IMAGE_FIELDS = [
    "image_id",
    "reference_canopy_pixels",
    "baseline_canopy_pixels",
    "proposed_canopy_pixels",
    "reference_temp_c",
    "baseline_temp_c",
    "proposed_temp_c",
    "baseline_error_c",
    "proposed_error_c",
    "baseline_abs_error_c",
    "proposed_abs_error_c",
    "baseline_valid",
    "proposed_valid",
]
SUMMARY_FIELDS = [
    "method",
    "n_total",
    "n_valid",
    "n_invalid",
    "mae_c",
    "rmse_c",
    "bias_c",
]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_temperature_matrix(path):
    values = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            parts = line.strip().split("=", 1)
            if len(parts) != 2:
                continue
            try:
                index = int(
                    parts[0].strip().replace("output[", "").replace("]", "")
                )
                value = float(parts[1].strip())
            except ValueError:
                continue
            values.append((index, value))

    if not values:
        raise ValueError("温度矩阵文件为空: {}".format(path))
    values.sort(key=lambda item: item[0])
    indexes = [item[0] for item in values]
    expected_indexes = list(range(384 * 512))
    if indexes != expected_indexes:
        raise ValueError("温度矩阵索引不完整或存在重复: {}".format(path))

    matrix = np.asarray(
        [item[1] for item in values], dtype=np.float32
    ).reshape((384, 512))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("温度矩阵包含非有限值: {}".format(path))
    return matrix


def read_test_ids(path):
    image_ids = [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(image_ids) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            "test split 图像数不是 {}: {}".format(EXPECTED_IMAGE_COUNT, len(image_ids))
        )
    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError("test split 存在重复 image_id")
    return image_ids


def read_mask(path, allowed_values, expected_shape):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError("无法读取 mask: {}".format(path))
    if mask.ndim != 2:
        raise ValueError("mask 不是二维单通道: {}".format(path))
    if mask.shape != expected_shape:
        raise ValueError(
            "mask 尺寸不匹配: {}，mask={}，temperature={}".format(
                path, mask.shape, expected_shape
            )
        )
    values = {int(value) for value in np.unique(mask)}
    if not values.issubset(allowed_values):
        raise ValueError("mask 含非法像素值 {}: {}".format(sorted(values), path))
    return mask.astype(np.uint8, copy=False)


def canopy_mean_temperature(temperature, mask):
    if temperature.shape != mask.shape:
        raise ValueError(
            "温度矩阵与mask尺寸不匹配: {} != {}".format(
                temperature.shape, mask.shape
            )
        )
    canopy = mask == 1
    canopy_pixels = int(np.count_nonzero(canopy))
    if canopy_pixels == 0:
        return None, 0
    mean_temperature = float(np.mean(temperature[canopy], dtype=np.float64))
    return mean_temperature, canopy_pixels


def summarize_errors(reference, estimate):
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("参照温度与估计温度数量不一致")

    valid = np.isfinite(reference) & np.isfinite(estimate)
    n_total = int(reference.size)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise RuntimeError("没有可用于温度误差统计的有效图像")

    errors = estimate[valid] - reference[valid]
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_total - n_valid,
        "mae_c": float(np.mean(np.abs(errors), dtype=np.float64)),
        "rmse_c": float(np.sqrt(np.mean(np.square(errors), dtype=np.float64))),
        "bias_c": float(np.mean(errors, dtype=np.float64)),
    }


def _format_float(value):
    return "" if value is None else "{:.6f}".format(value)


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_scatter_plot(rows, png_path, svg_path):
    series = [
        ("Baseline", "baseline_temp_c", "#4C78A8"),
        ("Proposed", "proposed_temp_c", "#E45756"),
    ]
    all_values = []
    plot_data = []

    for title, key, color in series:
        reference = np.asarray(
            [row["reference_temp_c"] for row in rows], dtype=np.float64
        )
        estimate = np.asarray(
            [
                np.nan if row[key] is None else row[key]
                for row in rows
            ],
            dtype=np.float64,
        )
        valid = np.isfinite(reference) & np.isfinite(estimate)
        if not np.any(valid):
            raise RuntimeError("{}没有可绘图的有效温度结果".format(title))
        plot_data.append((title, reference[valid], estimate[valid], color))
        all_values.extend(reference[valid].tolist())
        all_values.extend(estimate[valid].tolist())

    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    margin = max((upper - lower) * 0.05, 0.25)
    limits = (lower - margin, upper + margin)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharex=True, sharey=True)
    for axis, (title, reference, estimate, color) in zip(axes, plot_data):
        axis.scatter(
            reference,
            estimate,
            s=22,
            alpha=0.75,
            color=color,
            edgecolors="none",
        )
        axis.plot(limits, limits, linestyle="--", linewidth=1.2, color="#444444")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("Reference canopy temperature (°C)")
        axis.grid(alpha=0.2, linewidth=0.6)
    axes[0].set_ylabel("Estimated canopy temperature (°C)")

    figure.tight_layout()
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)


def self_check():
    temperature = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    gt_mask = np.asarray([[1, 255], [0, 1]], dtype=np.uint8)
    pred_mask = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)

    reference, reference_pixels = canopy_mean_temperature(temperature, gt_mask)
    estimate, estimate_pixels = canopy_mean_temperature(temperature, pred_mask)
    assert reference_pixels == 2 and math.isclose(reference, 25.0)
    assert estimate_pixels == 2 and math.isclose(estimate, 15.0)

    summary = summarize_errors([25.0, 30.0], [15.0, 35.0])
    assert summary["n_valid"] == 2 and summary["n_invalid"] == 0
    assert math.isclose(summary["mae_c"], 7.5)
    assert math.isclose(summary["rmse_c"], math.sqrt(62.5))
    assert math.isclose(summary["bias_c"], -2.5)

    invalid_summary = summarize_errors([25.0, 30.0], [15.0, np.nan])
    assert invalid_summary["n_valid"] == 1
    assert invalid_summary["n_invalid"] == 1
    print("[SelfCheck] PASS")


def main(
    test_split,
    temperature_dir,
    mask_dir,
    baseline_pred_dir,
    proposed_pred_dir,
    output_dir,
):
    for path in (
        temperature_dir,
        mask_dir,
        baseline_pred_dir,
        proposed_pred_dir,
    ):
        if not path.is_dir():
            raise FileNotFoundError("输入目录不存在: {}".format(path))
    if output_dir.exists():
        raise FileExistsError("输出目录已存在，拒绝覆盖: {}".format(output_dir))

    image_ids = read_test_ids(test_split)
    rows = []

    for image_id in image_ids:
        temperature = parse_temperature_matrix(
            temperature_dir / (image_id + ".txt")
        )
        gt_mask = read_mask(
            mask_dir / (image_id + ".png"), {0, 1, 255}, temperature.shape
        )
        baseline_mask = read_mask(
            baseline_pred_dir / (image_id + ".png"), {0, 1}, temperature.shape
        )
        proposed_mask = read_mask(
            proposed_pred_dir / (image_id + ".png"), {0, 1}, temperature.shape
        )

        reference_temp, reference_pixels = canopy_mean_temperature(
            temperature, gt_mask
        )
        if reference_temp is None:
            raise RuntimeError("人工掩膜无冠层像素: {}".format(image_id))

        baseline_temp, baseline_pixels = canopy_mean_temperature(
            temperature, baseline_mask
        )
        proposed_temp, proposed_pixels = canopy_mean_temperature(
            temperature, proposed_mask
        )
        baseline_error = (
            None if baseline_temp is None else baseline_temp - reference_temp
        )
        proposed_error = (
            None if proposed_temp is None else proposed_temp - reference_temp
        )

        rows.append({
            "image_id": image_id,
            "reference_canopy_pixels": reference_pixels,
            "baseline_canopy_pixels": baseline_pixels,
            "proposed_canopy_pixels": proposed_pixels,
            "reference_temp_c": reference_temp,
            "baseline_temp_c": baseline_temp,
            "proposed_temp_c": proposed_temp,
            "baseline_error_c": baseline_error,
            "proposed_error_c": proposed_error,
            "baseline_abs_error_c": (
                None if baseline_error is None else abs(baseline_error)
            ),
            "proposed_abs_error_c": (
                None if proposed_error is None else abs(proposed_error)
            ),
            "baseline_valid": int(baseline_temp is not None),
            "proposed_valid": int(proposed_temp is not None),
        })

    references = [row["reference_temp_c"] for row in rows]
    summaries = []
    for method, key in (
        ("baseline", "baseline_temp_c"),
        ("proposed", "proposed_temp_c"),
    ):
        summary = summarize_errors(references, [
            np.nan if row[key] is None else row[key] for row in rows
        ])
        summaries.append({"method": method, **summary})

    output_dir.mkdir(parents=True)
    per_image_rows = [
        {
            **row,
            **{
                key: _format_float(row[key])
                for key in (
                    "reference_temp_c",
                    "baseline_temp_c",
                    "proposed_temp_c",
                    "baseline_error_c",
                    "proposed_error_c",
                    "baseline_abs_error_c",
                    "proposed_abs_error_c",
                )
            },
        }
        for row in rows
    ]
    summary_rows = [
        {
            **summary,
            "mae_c": _format_float(summary["mae_c"]),
            "rmse_c": _format_float(summary["rmse_c"]),
            "bias_c": _format_float(summary["bias_c"]),
        }
        for summary in summaries
    ]

    _write_csv(
        output_dir / "per_image_temperature.csv",
        PER_IMAGE_FIELDS,
        per_image_rows,
    )
    _write_csv(
        output_dir / "summary_temperature_metrics.csv",
        SUMMARY_FIELDS,
        summary_rows,
    )
    save_scatter_plot(
        rows,
        output_dir / "reference_vs_estimated_temperature.png",
        output_dir / "reference_vs_estimated_temperature.svg",
    )

    print("[Temperature] 完成 {} 幅图像统计".format(len(rows)))
    for summary in summaries:
        print(
            "[Temperature] {method}: n={n_valid}, invalid={n_invalid}, "
            "MAE={mae_c:.6f}℃, RMSE={rmse_c:.6f}℃, Bias={bias_c:.6f}℃".format(
                **summary
            )
        )
    print("[Temperature] 产物目录: {}".format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maize canopy temperature extraction and statistics")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a synthetic self-check without project data or analysis outputs",
    )
    parser.add_argument(
        "--test-split",
        default=str(TEST_SPLIT),
    )
    parser.add_argument(
        "--temperature-dir",
        default=str(TEMPERATURE_DIR),
    )
    parser.add_argument(
        "--mask-dir",
        default=str(MASK_DIR),
    )
    parser.add_argument(
        "--baseline-pred-dir",
        default=str(BASELINE_PRED_DIR),
        help="Directory containing baseline prediction masks",
    )
    parser.add_argument(
        "--proposed-pred-dir",
        default=str(PROPOSED_PRED_DIR),
        help="Directory containing proposed-method prediction masks",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Analysis output directory; must not already exist",
    )
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        main(
            resolve_path(args.test_split),
            resolve_path(args.temperature_dir),
            resolve_path(args.mask_dir),
            resolve_path(args.baseline_pred_dir),
            resolve_path(args.proposed_pred_dir),
            resolve_path(args.output_dir),
        )
