from pathlib import Path
import pandas as pd


BASE_DIR = Path("outputs/experiments")
OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)

OUT_TEX = OUT_DIR / "fig_f1_evolution_all_models.tex"
OUT_CSV = OUT_DIR / "f1_evolution_all_models.csv"

MODELS = {
    "groq__compound_mini": {
        "label": "compound-mini",
        "color": "orange",
        "mark": "*",
    },
    "openai__gpt_oss_20b": {
        "label": "gpt-oss-20b",
        "color": "blue",
        "mark": "square*",
    },
    "groq__compound": {
        "label": "compound",
        "color": "green!60!black",
        "mark": "triangle*",
    },
    "qwen__qwen3_32b": {
        "label": "qwen3-32b",
        "color": "purple",
        "mark": "diamond*",
    },
    "llama_3_1_8b_instant": {
        "label": "llama3.1-8b",
        "color": "red",
        "mark": "pentagon*",
    },
    "llama_3_3_70b_versatile": {
        "label": "llama3.3-70b",
        "color": "cyan!70!black",
        "mark": "otimes*",
    },
    "openai__gpt_oss_120b": {
        "label": "gpt-oss-120b",
        "color": "gray",
        "mark": "star",
    },
}


def read_model_metrics(folder: str) -> pd.DataFrame:
    path = BASE_DIR / folder / "metrics_history.csv"

    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame(columns=["iteration", "f1_score_masquerade"])

    df = pd.read_csv(path)

    df = df[df["evaluation_type"] == "variant_external_test"].copy()

    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["f1_score_masquerade"] = pd.to_numeric(
        df["f1_score_masquerade"],
        errors="coerce",
    )

    df = df.dropna(subset=["iteration", "f1_score_masquerade"])
    df["iteration"] = df["iteration"].astype(int)

    return df[["iteration", "f1_score_masquerade"]].sort_values("iteration")


def format_coordinates(df: pd.DataFrame) -> str:
    lines = []

    for _, row in df.iterrows():
        iteration = int(row["iteration"])
        f1 = float(row["f1_score_masquerade"])
        lines.append(f"        ({iteration},{f1:.4f})")

    return "\n".join(lines)


def build_chart() -> str:
    plots = []

    for folder, meta in MODELS.items():
        df = read_model_metrics(folder)

        if df.empty:
            continue

        coords = format_coordinates(df)

        plots.append(
f"""
\\addplot[
    color={meta["color"]},
    mark={meta["mark"]},
    thick
]
coordinates {{
{coords}
}};
\\addlegendentry{{\\texttt{{{meta["label"]}}}}}
"""
        )

    plots_text = "\n".join(plots)

    return f"""
\\begin{{figure*}}[ht]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=0.95\\textwidth,
    height=7cm,
    xlabel={{Attack Variant}},
    ylabel={{F1-score}},
    xmin=0.5,
    xmax=30.5,
    ymin=0.90,
    ymax=1.01,
    xtick={{1,5,10,15,20,25,30}},
    xticklabels={{v1,v5,v10,v15,v20,v25,v30}},
    ytick={{0.90,0.92,0.94,0.96,0.98,1.00}},
    grid=major,
    grid style=dashed,
    thick,
    legend style={{
        at={{(0.5,-0.25)}},
        anchor=north,
        legend columns=3,
        font=\\small
    }},
    mark size=1.6pt
]
{plots_text}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{F1-score evolution across adversarial attack variants for all LLM-based agents. Only variants effectively generated and evaluated are plotted; skipped iterations are omitted. Lower values indicate stronger degradation of the baseline-trained IDS.}}
\\label{{fig:f1_evolution_all_models}}
\\end{{figure*}}
"""


def build_wide_csv() -> pd.DataFrame:
    wide = pd.DataFrame({"iteration": list(range(1, 31))})

    for folder, meta in MODELS.items():
        df = read_model_metrics(folder)

        if df.empty:
            continue

        df = df.rename(columns={"f1_score_masquerade": meta["label"]})
        wide = wide.merge(df, on="iteration", how="left")

    return wide


def main() -> None:
    chart = build_chart()
    OUT_TEX.write_text(chart, encoding="utf-8")

    wide = build_wide_csv()
    wide.to_csv(OUT_CSV, index=False)

    print(f"[OK] LaTeX figure saved to: {OUT_TEX}")
    print(f"[OK] CSV table saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()