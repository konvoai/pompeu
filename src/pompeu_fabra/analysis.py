"""Generate aggregate analysis and figures from judgement JSON files.

This module loads the evaluation judgements, aggregates the scoring metrics,
persists tabular summaries, and renders visualisations into the `analysis`
directory. It is intended to be run via `uv run analysis`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
JUDGEMENTS_DIR = ROOT / "judgements"
ANALYSIS_DIR = ROOT / "analysis"
METRIC_KEYS = ("quality", "correctness", "grammar", "completeness")
METRIC_STYLES = {
    "quality": {"color": "#3B73B9", "title": "Average quality score per model"},
    "correctness": {"color": "#2CA25F", "title": "Average correctness score per model"},
    "grammar": {"color": "#D95F02", "title": "Average grammar score per model"},
    "completeness": {"color": "#756BB1", "title": "Average completeness score per model"},
}
LATENCY_STYLE = {
    "color": "#5E5E5E",
    "title": "Average latency per model (per message)",
    "xlabel": "Average latency per message (seconds — lower is better)",
}
GRAMMAR_LATENCY_STYLE = {
    "title": "Grammar vs latency (per message)",
    "xlabel": "Average latency per message (seconds — lower is better)",
    "ylabel": "Average grammar score (higher is better)",
    "highlight_color": "#D95F02",
    "point_color": "#6BAED6",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_judgements(judgements_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for json_path in sorted(judgements_dir.glob("*.json")):
        data = _load_json(json_path)
        record: Dict[str, Any] = {
            "id": data.get("id"),
            "model": data.get("modelName"),
            "goal": data.get("goal"),
            "start_time": data.get("startTime"),
            "end_time": data.get("endTime"),
        }

        conversation = list(data.get("conversation", []))
        record["conversation_turns"] = len(conversation)
        record["conversation_tokens"] = sum(
            len(str(message.get("message", "")).split()) for message in conversation
        )

        for metric in METRIC_KEYS:
            metric_data = data.get(metric, {})
            record[f"{metric}_score"] = metric_data.get("score")

        metric_scores = [
            record.get(f"{metric}_score")
            for metric in METRIC_KEYS
            if record.get(f"{metric}_score") is not None
        ]
        record["overall_score"] = (
            sum(metric_scores) / len(metric_scores) if metric_scores else None
        )

        if (
            isinstance(record.get("start_time"), (int, float))
            and isinstance(record.get("end_time"), (int, float))
            and record["end_time"] >= record["start_time"]
        ):
            record["latency_seconds"] = (record["end_time"] - record["start_time"]) / 1000
        else:
            record["latency_seconds"] = None

        turns = record.get("conversation_turns") or 0
        if record["latency_seconds"] is not None and turns > 0:
            record["latency_seconds_per_message"] = record["latency_seconds"] / turns
        else:
            record["latency_seconds_per_message"] = None

        records.append(record)

    if not records:
        raise FileNotFoundError(f"No judgement JSON files found in {judgements_dir}")

    return pd.DataFrame(records)


def compute_model_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        [f"{metric}_score" for metric in METRIC_KEYS]
        + ["overall_score", "latency_seconds", "latency_seconds_per_message"]
    )

    def _rename(name: str) -> str:
        if name.endswith("_score"):
            return name.replace("_score", "_avg")
        if name == "latency_seconds":
            return "latency_seconds_avg"
        if name == "latency_seconds_per_message":
            return "latency_seconds_per_message_avg"
        return name
    grouped = (
        df.groupby("model")[metric_columns]
        .mean()
        .rename(columns=_rename)
    )
    grouped["judgement_count"] = df.groupby("model").size()
    return grouped.sort_values("overall_avg", ascending=False)


def filter_scored_models(per_model: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [f"{metric}_avg" for metric in METRIC_KEYS]
    scores_sum = per_model[metric_columns].fillna(0).sum(axis=1)
    filtered = per_model[scores_sum > 0].copy()
    if filtered.empty:
        raise ValueError("All models were filtered out due to missing scores.")
    return filtered


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_tables(df: pd.DataFrame, per_model: pd.DataFrame) -> None:
    df.sort_values(["model", "id"]).to_csv(
        ANALYSIS_DIR / "judgements_flat.csv", index=False
    )
    per_model.to_csv(ANALYSIS_DIR / "metrics_by_model.csv")


def plot_metric_bar(per_model: pd.DataFrame, metric: str) -> Path:
    if metric not in METRIC_STYLES:
        raise ValueError(f"Unknown metric '{metric}'")

    style = METRIC_STYLES[metric]
    column = f"{metric}_avg"
    chart_data = per_model.reset_index().sort_values(column, ascending=True)
    palette = sns.light_palette(style["color"], n_colors=len(chart_data) or 1, reverse=False)

    figure_path = ANALYSIS_DIR / f"{metric}_avg_by_model.png"
    plt.figure(figsize=(11, max(4, 0.65 * len(chart_data))))
    ax = sns.barplot(
        data=chart_data,
        x=column,
        y="model",
        order=chart_data["model"].tolist(),
        hue="model",
        hue_order=chart_data["model"].tolist(),
        palette=palette,
        edgecolor="black",
        legend=False,
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"Average {metric.capitalize()} score")
    ax.set_ylabel("Model")
    ax.set_title(style["title"])

    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        label_x = min(1.02, width + 0.015)
        ax.text(label_x, y, f"{width:.3f}", va="center", ha="left", fontsize=9)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()
    return figure_path


def plot_latency_bar(per_model: pd.DataFrame) -> Path:
    column = "latency_seconds_per_message_avg"
    if column not in per_model.columns:
        raise ValueError("Latency averages are missing from aggregates.")

    chart_data = (
        per_model.reset_index()[["model", column]]
        .dropna(subset=[column])
        .sort_values(column, ascending=True)
    )
    if chart_data.empty:
        raise ValueError("No latency data available to plot.")
    figure_path = ANALYSIS_DIR / "average_latency_by_model.png"

    plt.figure(figsize=(11, max(4, 0.65 * len(chart_data))))
    ax = sns.barplot(
        data=chart_data,
        x=column,
        y="model",
        order=chart_data["model"].tolist(),
        hue="model",
        hue_order=chart_data["model"].tolist(),
        palette=sns.light_palette(LATENCY_STYLE["color"], n_colors=len(chart_data) or 1),
        edgecolor="black",
        legend=False,
    )

    max_latency = float(chart_data[column].max())
    ax.set_xlim(0, max(1.0, max_latency * 1.05))
    ax.set_xlabel(LATENCY_STYLE["xlabel"])
    ax.set_ylabel("Model")
    ax.set_title(LATENCY_STYLE["title"])

    label_offset = max(0.02 * max_latency, 0.02)
    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(
            width + label_offset,
            y,
            f"{width:.2f}s/msg",
            va="center",
            ha="left",
            fontsize=9,
        )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()
    return figure_path


def plot_grammar_latency_scatter(per_model: pd.DataFrame) -> Path:
    grammar_column = "grammar_avg"
    latency_column = "latency_seconds_per_message_avg"

    missing_columns = [
        column
        for column in (grammar_column, latency_column)
        if column not in per_model.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for grammar vs latency plot: {missing_columns}"
        )

    chart_data = (
        per_model.reset_index()[["model", grammar_column, latency_column]]
        .dropna(subset=[grammar_column, latency_column])
        .sort_values(grammar_column, ascending=False)
    )
    if chart_data.empty:
        raise ValueError("No combined grammar and latency data available to plot.")

    chart_data["grammar_rank"] = chart_data[grammar_column].rank(
        ascending=False, method="dense"
    )
    chart_data["latency_rank"] = chart_data[latency_column].rank(
        ascending=True, method="dense"
    )
    chart_data["combined_rank"] = chart_data["grammar_rank"] + chart_data["latency_rank"]
    best_row = chart_data.nsmallest(1, "combined_rank").iloc[0]

    figure_path = ANALYSIS_DIR / "grammar_vs_latency.png"
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=chart_data,
        x=latency_column,
        y=grammar_column,
        s=110,
        color=GRAMMAR_LATENCY_STYLE["point_color"],
        edgecolor="black",
    )

    ax.scatter(
        best_row[latency_column],
        best_row[grammar_column],
        s=160,
        color=GRAMMAR_LATENCY_STYLE["highlight_color"],
        edgecolor="black",
        zorder=5,
        label=f"Best trade-off: {best_row['model']}",
    )

    for _, row in chart_data.iterrows():
        ax.text(
            row[latency_column],
            row[grammar_column] + 0.008,
            row["model"],
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(GRAMMAR_LATENCY_STYLE["title"])
    ax.set_xlabel(GRAMMAR_LATENCY_STYLE["xlabel"])
    ax.set_ylabel(GRAMMAR_LATENCY_STYLE["ylabel"])
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=220)
    plt.close()
    return figure_path


def identify_metric_leaders(per_model: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    leaders: Dict[str, Dict[str, Any]] = {}
    for metric in METRIC_KEYS:
        column = f"{metric}_avg"
        top_index = per_model[column].idxmax()
        leaders[metric] = {
            "model": top_index,
            "score": float(per_model.loc[top_index, column]),
        }
    return leaders


def identify_latency_leader(per_model: pd.DataFrame) -> Dict[str, Any]:
    column = "latency_seconds_per_message_avg"
    if column not in per_model.columns:
        return {}
    valid = per_model[column].dropna()
    if valid.empty:
        return {}
    fastest_index = valid.idxmin()
    return {
        "model": fastest_index,
        "seconds_per_message": float(per_model.loc[fastest_index, column]),
    }


def build_summary(per_model: pd.DataFrame) -> Dict[str, Any]:
    top_row = per_model.iloc[0]
    return {
        "top_model": per_model.index[0],
        "top_model_overall_avg": float(top_row["overall_avg"]),
        "judgement_counts": {
            model: int(count) for model, count in per_model["judgement_count"].items()
        },
        "metric_leaders": identify_metric_leaders(per_model),
        "latency_leader": identify_latency_leader(per_model),
    }


def write_summary(summary: Dict[str, Any]) -> None:
    summary_path = ANALYSIS_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    sns.set_theme(style="whitegrid")
    ensure_output_dir(ANALYSIS_DIR)

    dataframe = load_judgements(JUDGEMENTS_DIR)
    per_model_all = compute_model_aggregates(dataframe)
    per_model = filter_scored_models(per_model_all)

    export_tables(dataframe, per_model)

    figure_paths = [plot_metric_bar(per_model, metric) for metric in METRIC_KEYS]
    figure_paths.append(plot_latency_bar(per_model))
    try:
        figure_paths.append(plot_grammar_latency_scatter(per_model))
    except ValueError as error:
        print(f"Skipping grammar vs latency plot: {error}")
    summary = build_summary(per_model)
    write_summary(summary)

    print("Analysis completed.")
    print(f"Top model: {summary['top_model']} (overall avg {summary['top_model_overall_avg']:.3f})")
    print("Metric leaders:")
    for metric, data in summary["metric_leaders"].items():
        print(f" - {metric}: {data['model']} ({data['score']:.3f})")
    latency_leader = summary.get("latency_leader") or {}
    if latency_leader:
        print(
            "Fastest model (per message): "
            f"{latency_leader['model']} ({latency_leader['seconds_per_message']:.2f}s average latency per message)"
        )
    print("Generated figures:")
    for path in figure_paths:
        print(f" - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

