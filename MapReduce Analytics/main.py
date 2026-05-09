import argparse
import csv
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Este script necesita pyarrow para leer los archivos Parquet generados "
        "en la parte 1. No usa Pandas ni librerias de procesamiento distribuido."
    ) from exc


STOPWORDS = {
    "a", "actualmente", "ademas", "ahi", "al", "algo", "algunas", "algunos",
    "ante", "antes", "aquel", "aquella", "aquellas", "aquello", "aquellos",
    "aqui", "asi", "aunque", "ayer", "bajo", "bien", "cada", "casi", "como",
    "con", "contra", "cual", "cuando", "cuenta", "da", "de", "debe", "debido",
    "del", "desde", "despues", "dia", "dias", "dice", "dijo", "donde", "dos",
    "durante", "e", "el", "ella", "ellas", "ello", "ellos", "embargo", "en",
    "entre", "era", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estaba", "estado", "estan", "estar", "estas", "este", "esto", "estos",
    "fue", "fueron", "ha", "habia", "han", "hasta", "hay", "hizo", "hoy",
    "la", "las", "le", "les", "lo", "los", "luego", "mas", "me", "mediante",
    "menos", "mi", "mientras", "mis", "muy", "no", "nos", "nuevo", "nueva",
    "nuevas", "nuevos", "o", "otra", "otras", "otro", "otros", "para", "parte",
    "pero", "poco", "por", "porque", "puede", "pueden", "que", "quien",
    "quienes", "realizo", "se", "sea", "segun", "ser", "sera", "si", "sin",
    "solo", "sobre", "son", "su", "sus", "tambien", "tanto", "te", "tener",
    "tiene", "tienen", "todo", "todos", "tras", "tu", "un", "una", "unas",
    "uno", "unos", "vez", "y", "ya",
}

TOKEN_RE = re.compile(r"[a-z]+")
UNKNOWN_REGION_ID = 17
MIN_TOKEN_LENGTH = 4
MAX_TOKEN_LENGTH = 24

DOMAIN_STOPWORDS = {
    "ano", "anos", "abril", "agosto", "chile", "chilena", "chilenas",
    "chileno", "chilenos", "diciembre", "enero", "febrero", "julio",
    "junio", "marzo", "mayo", "nacional", "noviembre", "octubre", "pais",
    "personas", "septiembre",
}
STOPWORDS.update(DOMAIN_STOPWORDS)


def normalize_text(text):
    text = "" if text is None else str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def tokenize(*texts):
    normalized = normalize_text(" ".join("" if text is None else str(text) for text in texts))
    for token in TOKEN_RE.findall(normalized):
        if is_interpretable_token(token):
            yield token


def is_interpretable_token(token):
    if token in STOPWORDS:
        return False
    if len(token) < MIN_TOKEN_LENGTH or len(token) > MAX_TOKEN_LENGTH:
        return False
    if len(set(token)) <= 2:
        return False
    return True


def fact_files(warehouse_dir):
    fact_dir = Path(warehouse_dir) / "fact_news"
    if not fact_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de hechos: {fact_dir}")
    return sorted(path for path in fact_dir.rglob("*.parquet") if path.is_file())


def iter_fact_rows(warehouse_dir, columns=None, batch_size=50000):
    for file_path in fact_files(warehouse_dir):
        parquet_file = pq.ParquetFile(file_path)
        available_columns = set(parquet_file.schema.names)
        selected_columns = [col for col in (columns or available_columns) if col in available_columns]
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=selected_columns):
            for row in batch.to_pylist():
                yield row


def read_dimension(warehouse_dir, dim_name, key_column, value_column):
    dim_path = Path(warehouse_dir) / dim_name / f"{dim_name}.parquet"
    if not dim_path.exists():
        return {}
    table = pq.read_table(dim_path, columns=[key_column, value_column])
    return {row[key_column]: row[value_column] for row in table.to_pylist()}


def yyyymm_from_row(row):
    if row.get("year") is not None and row.get("month") is not None:
        return f"{int(row['year']):04d}-{int(row['month']):02d}"
    date_id = int(row["date_id"])
    return f"{date_id // 10000:04d}-{(date_id // 100) % 100:02d}"


def date_from_row(row):
    date_id = int(row["date_id"])
    return f"{date_id // 10000:04d}-{(date_id // 100) % 100:02d}-{date_id % 100:02d}"


def map_reduce_sum(records, mapper):
    reduced = Counter()
    for record in records:
        for key, value in mapper(record):
            reduce_sum(reduced, (key, value))
    return reduced


def reduce_sum(accumulator, mapped_pair):
    key, value = mapped_pair
    accumulator[key] += value
    return accumulator


def monthly_term_mapper(row):
    month = yyyymm_from_row(row)
    for token in tokenize(row.get("title"), row.get("body")):
        yield (month, token), 1


def top_k_monthly_terms(warehouse_dir, top_k):
    columns = ["title", "body", "date_id", "year", "month"]
    counts = map_reduce_sum(iter_fact_rows(warehouse_dir, columns=columns), monthly_term_mapper)
    by_month = defaultdict(Counter)
    for (month, token), count in counts.items():
        by_month[month][token] = count
    return [
        (month, token, count)
        for month in sorted(by_month)
        for token, count in by_month[month].most_common(top_k)
    ]


def global_word_mapper(row):
    for token in tokenize(row.get("title"), row.get("body")):
        yield token, 1


def regional_word_mapper(row, exclude_unknown_region):
    region_id = int(row["region_id"])
    if exclude_unknown_region and region_id == UNKNOWN_REGION_ID:
        return
    for token in tokenize(row.get("title"), row.get("body")):
        yield (region_id, token), 1


def region_token_total_mapper(row, exclude_unknown_region):
    region_id = int(row["region_id"])
    if exclude_unknown_region and region_id == UNKNOWN_REGION_ID:
        return
    total = sum(1 for _ in tokenize(row.get("title"), row.get("body")))
    if total:
        yield region_id, total


def regional_word_distribution(
    warehouse_dir,
    min_global_count,
    min_regional_count,
    exclude_unknown_region,
    limit_per_region,
):
    columns = ["title", "body", "region_id"]
    global_counts = map_reduce_sum(iter_fact_rows(warehouse_dir, columns=columns), global_word_mapper)
    regional_counts = map_reduce_sum(
        iter_fact_rows(warehouse_dir, columns=columns),
        lambda row: regional_word_mapper(row, exclude_unknown_region),
    )
    region_totals = map_reduce_sum(
        iter_fact_rows(warehouse_dir, columns=columns),
        lambda row: region_token_total_mapper(row, exclude_unknown_region),
    )

    global_total = sum(global_counts.values())
    rows_by_region = defaultdict(list)
    for (region_id, token), regional_count in regional_counts.items():
        global_count = global_counts[token]
        if global_count < min_global_count or regional_count < min_regional_count:
            continue
        regional_relative = regional_count / region_totals[region_id]
        global_relative = global_count / global_total
        ratio = regional_relative / global_relative if global_relative else 0.0
        rows_by_region[region_id].append((region_id, token, regional_count, global_count, ratio))

    output = []
    for region_id in sorted(rows_by_region):
        rows = sorted(rows_by_region[region_id], key=lambda item: (-item[4], -item[2], item[1]))
        if limit_per_region > 0:
            rows = rows[:limit_per_region]
        output.extend(rows)
    return output


def source_word_mapper(row):
    source_id = int(row["source_id"])
    for token in tokenize(row.get("title"), row.get("body")):
        yield (source_id, token), 1


def source_vocabulary_divergence(warehouse_dir, min_global_count, min_source_terms):
    columns = ["title", "body", "source_id"]
    global_counts = map_reduce_sum(iter_fact_rows(warehouse_dir, columns=columns), global_word_mapper)
    source_counts = map_reduce_sum(iter_fact_rows(warehouse_dir, columns=columns), source_word_mapper)

    allowed_tokens = {token for token, count in global_counts.items() if count >= min_global_count}
    global_total = sum(global_counts[token] for token in allowed_tokens)
    if global_total == 0:
        return []
    source_totals = Counter()
    source_vocab = defaultdict(Counter)

    for (source_id, token), count in source_counts.items():
        if token not in allowed_tokens:
            continue
        source_totals[source_id] += count
        source_vocab[source_id][token] = count

    rows = []
    for source_id in sorted(source_vocab):
        total = source_totals[source_id]
        if total < min_source_terms:
            continue
        kl_divergence = 0.0
        for token, count in source_vocab[source_id].items():
            source_probability = count / total
            global_probability = global_counts[token] / global_total
            kl_divergence += source_probability * math.log(source_probability / global_probability)
        rows.append((source_id, total, len(source_vocab[source_id]), kl_divergence))
    return sorted(rows, key=lambda item: item[3], reverse=True)


def daily_count_mapper(row):
    yield date_from_row(row), 1


def detect_daily_peaks(warehouse_dir, window_size, factor, min_delta):
    columns = ["date_id"]
    daily_counts = map_reduce_sum(iter_fact_rows(warehouse_dir, columns=columns), daily_count_mapper)
    if not daily_counts:
        return []

    start = date.fromisoformat(min(daily_counts))
    end = date.fromisoformat(max(daily_counts))
    dates = []
    current = start
    while current <= end:
        dates.append(current.isoformat())
        current += timedelta(days=1)

    rows = []

    for index, day in enumerate(dates):
        history = dates[max(0, index - window_size):index]
        moving_average = (
            sum(daily_counts[history_date] for history_date in history) / len(history)
            if history else 0.0
        )
        count = daily_counts[day]
        is_peak = bool(
            moving_average > 0
            and count >= moving_average * factor
            and count - moving_average >= min_delta
        )
        rows.append((day, count, moving_average, count - moving_average, is_peak))
    return rows


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def write_peak_svg(path, daily_rows, width=1200, height=420):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not daily_rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return

    padding_left, padding_right, padding_top, padding_bottom = 70, 30, 30, 60
    chart_width = width - padding_left - padding_right
    chart_height = height - padding_top - padding_bottom
    max_count = max(max(row[1], row[2]) for row in daily_rows) or 1

    def x_pos(index):
        return padding_left + (chart_width * index / max(1, len(daily_rows) - 1))

    def y_pos(value):
        return padding_top + chart_height - (chart_height * value / max_count)

    count_points = " ".join(
        f"{x_pos(index):.1f},{y_pos(count):.1f}"
        for index, (_, count, _, _, _) in enumerate(daily_rows)
    )
    average_points = " ".join(
        f"{x_pos(index):.1f},{y_pos(average):.1f}"
        for index, (_, _, average, _, _) in enumerate(daily_rows)
    )
    peak_circles = "\n".join(
        f"<circle cx=\"{x_pos(index):.1f}\" cy=\"{y_pos(count):.1f}\" r=\"4\" fill=\"#d62728\"><title>{date}: {count}</title></circle>"
        for index, (date, count, _, _, is_peak) in enumerate(daily_rows)
        if is_peak
    )

    start_date = daily_rows[0][0]
    end_date = daily_rows[-1][0]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <line x1="{padding_left}" y1="{padding_top + chart_height}" x2="{width - padding_right}" y2="{padding_top + chart_height}" stroke="#333"/>
  <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_height}" stroke="#333"/>
  <text x="{padding_left}" y="22" font-family="Arial" font-size="18" fill="#111">Volumen diario de articulos y peaks detectados</text>
  <polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{count_points}"/>
  <polyline fill="none" stroke="#ff7f0e" stroke-width="2" stroke-dasharray="6 4" points="{average_points}"/>
  {peak_circles}
  <text x="{padding_left}" y="{height - 20}" font-family="Arial" font-size="12" fill="#333">{start_date}</text>
  <text x="{width - padding_right - 80}" y="{height - 20}" font-family="Arial" font-size="12" fill="#333">{end_date}</text>
  <text x="{width - 300}" y="24" font-family="Arial" font-size="12" fill="#1f77b4">Conteo diario</text>
  <text x="{width - 190}" y="24" font-family="Arial" font-size="12" fill="#ff7f0e">Promedio movil</text>
  <text x="{width - 80}" y="24" font-family="Arial" font-size="12" fill="#d62728">Peak</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def add_names(rows, names, key_index, insert_after_index):
    named_rows = []
    for row in rows:
        key = row[key_index]
        name = names.get(key, f"id_{key}")
        named_rows.append(row[:insert_after_index + 1] + (name,) + row[insert_after_index + 1:])
    return named_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analitica MapReduce manual sobre el warehouse de noticias."
    )
    parser.add_argument("--warehouse", default="warehouse", help="Carpeta warehouse generada por la parte 1.")
    parser.add_argument("--output", default="mapreduce_output", help="Carpeta de resultados.")
    parser.add_argument("--top-k", type=int, default=20, help="Cantidad de terminos por mes.")
    parser.add_argument("--min-global-count", type=int, default=500, help="Frecuencia global minima para analisis regional/fuentes.")
    parser.add_argument("--min-regional-count", type=int, default=200, help="Frecuencia regional minima para analisis regional.")
    parser.add_argument("--min-source-terms", type=int, default=1000, help="Cantidad minima de terminos por fuente para divergencia.")
    parser.add_argument("--region-limit", type=int, default=0, help="Maximo de filas por region. 0 deja todas.")
    parser.add_argument("--include-unknown-region", action="store_true", help="Incluye region Desconocida en analisis regional.")
    parser.add_argument("--peak-window", type=int, default=7, help="Ventana historica para promedio movil diario.")
    parser.add_argument("--peak-factor", type=float, default=1.8, help="Factor sobre promedio movil para marcar peak.")
    parser.add_argument("--peak-min-delta", type=float, default=50.0, help="Diferencia minima contra promedio movil para marcar peak.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    warehouse_dir = Path(args.warehouse)

    print("Inicio analitica MapReduce...")

    monthly_top = top_k_monthly_terms(warehouse_dir, args.top_k)
    write_csv(
        output_dir / "top_terms_monthly.csv",
        ["year_month", "term", "count"],
        monthly_top,
    )
    print("2.1 Top-K terminos mensuales listo.")

    region_names = read_dimension(warehouse_dir, "dim_region", "region_key", "region_name")
    regional_rows = regional_word_distribution(
        warehouse_dir,
        args.min_global_count,
        args.min_regional_count,
        not args.include_unknown_region,
        args.region_limit,
    )
    regional_rows = add_names(regional_rows, region_names, key_index=0, insert_after_index=0)
    write_csv(
        output_dir / "regional_word_distribution.csv",
        ["region_id", "region_name", "term", "regional_count", "global_count", "relative_ratio"],
        regional_rows,
    )
    print("2.2 Distribucion de palabras por region lista.")

    source_names = read_dimension(warehouse_dir, "dim_source", "source_id", "source_name")
    source_rows = source_vocabulary_divergence(warehouse_dir, args.min_global_count, args.min_source_terms)
    source_rows = add_names(source_rows, source_names, key_index=0, insert_after_index=0)
    write_csv(
        output_dir / "source_vocabulary_divergence.csv",
        ["source_id", "source_name", "total_terms", "vocabulary_size", "kl_divergence"],
        source_rows,
    )
    print("2.3 Divergencia de vocabulario por fuente lista.")

    daily_rows = detect_daily_peaks(
        warehouse_dir,
        args.peak_window,
        args.peak_factor,
        args.peak_min_delta,
    )
    write_csv(
        output_dir / "daily_peaks.csv",
        ["date", "article_count", "moving_average", "delta", "is_peak"],
        daily_rows,
    )
    write_peak_svg(output_dir / "daily_volume_peaks.svg", daily_rows)
    print("2.4 Peaks diarios listos.")
    print(f"Resultados escritos en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
