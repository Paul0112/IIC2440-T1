import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Este script necesita matplotlib para generar los graficos. "
        "Instalalo con: pip install matplotlib"
    ) from exc


DEFAULT_REGIONS = {
    "Arica y Parinacota",
    "Tarapaca",
    "Valparaiso",
    "Metropolitana",
    "Biobio",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def save_current_figure(path):
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_global_terms(output_dir, figures_dir, top_n):
    rows = read_csv(output_dir / "top_terms_monthly.csv")
    totals = Counter()
    for row in rows:
        totals[row["term"]] += int(row["count"])

    data = totals.most_common(top_n)
    terms = [term for term, _ in data]
    counts = [count for _, count in data]

    plt.figure(figsize=(11, 6))
    plt.barh(terms[::-1], counts[::-1], color="#2563eb")
    plt.title("Terminos mas frecuentes en rankings mensuales")
    plt.xlabel("Ocurrencias acumuladas")
    plt.ylabel("Termino")
    save_current_figure(figures_dir / "global_top_terms.png")


def plot_monthly_leading_terms(output_dir, figures_dir):
    rows = read_csv(output_dir / "top_terms_monthly.csv")
    by_month = defaultdict(list)
    for row in rows:
        by_month[row["year_month"]].append((row["term"], int(row["count"])))

    months = sorted(by_month)
    leading_counts = []
    labels = []
    for month in months:
        term, count = max(by_month[month], key=lambda item: item[1])
        leading_counts.append(count)
        labels.append(term)

    x_values = list(range(len(months)))
    colors = ["#0f766e" if label == "chile" else "#64748b" for label in labels]

    plt.figure(figsize=(13, 6))
    plt.bar(x_values, leading_counts, color=colors)
    plt.title("Termino dominante por mes")
    plt.xlabel("Mes")
    plt.ylabel("Ocurrencias")
    plt.xticks(x_values, months, rotation=75, ha="right", fontsize=8)

    for index, label in enumerate(labels):
        plt.text(index, leading_counts[index], label, rotation=90, va="bottom", ha="center", fontsize=7)

    save_current_figure(figures_dir / "monthly_leading_terms.png")


def plot_source_divergence(output_dir, figures_dir, top_n):
    rows = read_csv(output_dir / "source_vocabulary_divergence.csv")
    rows = sorted(rows, key=lambda row: float(row["kl_divergence"]), reverse=True)[:top_n]

    names = [row["source_name"] for row in rows]
    divergences = [float(row["kl_divergence"]) for row in rows]

    plt.figure(figsize=(11, 6))
    plt.barh(names[::-1], divergences[::-1], color="#7c3aed")
    plt.title("Fuentes con mayor divergencia de vocabulario")
    plt.xlabel("Divergencia KL aproximada")
    plt.ylabel("Fuente")
    save_current_figure(figures_dir / "source_vocabulary_divergence.png")


def plot_daily_peaks(output_dir, figures_dir):
    rows = read_csv(output_dir / "daily_peaks.csv")
    dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in rows]
    counts = [int(row["article_count"]) for row in rows]
    moving_average = [float(row["moving_average"]) for row in rows]
    peaks = [row for row in rows if row["is_peak"] == "True"]
    peak_dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in peaks]
    peak_counts = [int(row["article_count"]) for row in peaks]

    plt.figure(figsize=(13, 6))
    plt.plot(dates, counts, color="#2563eb", linewidth=1.2, label="Articulos diarios")
    plt.plot(dates, moving_average, color="#f97316", linewidth=1.4, linestyle="--", label="Promedio movil")
    plt.scatter(peak_dates, peak_counts, color="#dc2626", s=42, label="Peak", zorder=5)
    plt.title("Volumen diario de noticias y peaks detectados")
    plt.xlabel("Fecha")
    plt.ylabel("Cantidad de articulos")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(figures_dir / "daily_peaks.png")


def plot_regional_terms(output_dir, figures_dir, regions, terms_per_region):
    rows = read_csv(output_dir / "regional_word_distribution.csv")
    grouped = defaultdict(list)

    for row in rows:
        region = row["region_name"]
        if region in regions:
            grouped[region].append(
                (
                    row["term"],
                    float(row["relative_ratio"]),
                    int(row["regional_count"]),
                )
            )

    for region in sorted(grouped):
        data = sorted(grouped[region], key=lambda item: (-item[2], -item[1], item[0]))[:terms_per_region]
        terms = [term for term, _, _ in data]
        ratios = [ratio for _, ratio, _ in data]

        safe_region = region.lower().replace(" ", "_").replace("/", "_")
        plt.figure(figsize=(10, 5))
        plt.barh(terms[::-1], ratios[::-1], color="#0891b2")
        plt.title(f"Terminos distintivos: {region}")
        plt.xlabel("Razon de frecuencia relativa")
        plt.ylabel("Termino")
        save_current_figure(figures_dir / f"regional_terms_{safe_region}.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera graficos a partir de los CSV producidos por la analitica MapReduce."
    )
    parser.add_argument("--output", default="MapReduce Analytics/output", help="Carpeta con CSV de resultados.")
    parser.add_argument("--figures", default="MapReduce Analytics/figures", help="Carpeta donde se guardan graficos.")
    parser.add_argument("--top-n", type=int, default=15, help="Cantidad de elementos en rankings generales.")
    parser.add_argument("--terms-per-region", type=int, default=10, help="Cantidad de terminos por grafico regional.")
    parser.add_argument(
        "--regions",
        nargs="*",
        default=sorted(DEFAULT_REGIONS),
        help="Regiones a graficar desde regional_word_distribution.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    figures_dir = Path(args.figures)

    ensure_dir(figures_dir)
    plot_global_terms(output_dir, figures_dir, args.top_n)
    plot_monthly_leading_terms(output_dir, figures_dir)
    plot_source_divergence(output_dir, figures_dir, args.top_n)
    plot_daily_peaks(output_dir, figures_dir)
    plot_regional_terms(output_dir, figures_dir, set(args.regions), args.terms_per_region)

    print(f"Graficos escritos en: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
