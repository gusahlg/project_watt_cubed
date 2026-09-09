#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Headless unit-throughput runner: one release lib build, then each pinned
# #[ignore] bench three times. Prints name | value | unit | pinned | delta %.
# New benches must be added to BENCHES below (cargo filter, print regex, unit,
# scale from the printed number to the doc-comment unit).

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

export CARGO_TERM_COLOR=never

RUNS=3
JSON_OUT=

usage() {
    cat <<'EOF'
Usage: scripts/bench-unit.sh [--json]

Build the release lib tests once, run each pinned #[ignore] throughput bench
three times, and print a median-vs-pinned table.

  --json   also write the same rows as JSON lines to benchmarks/unit.jsonl
EOF
}

for arg in "$@"; do
    case "$arg" in
        --json) JSON_OUT=benchmarks/unit.jsonl ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

# name # ere capturing the printed number # doc-comment unit # scale (printed → unit)
# generate_column_ms and admit_mesh_lane_20k_select are listed so they run when
# those tests exist on the branch; missing names are skipped.
BENCHES=(
    'mesh_result_channel_throughput#= ([0-9.]+) jobs/s#jobs/s#1'
    'padded_capture_throughput#= ([0-9.]+) captures/s#captures/s#1'
    'light_propagate_throughput#= ([0-9.]+) settles/s#settles/s#1'
    'generate_column_ms#([0-9.]+) ms/column#ms/column#1'
    'worldgen_column_cost#diffusion=([0-9.]+)ms#ms#1'
    'admit_mesh_lane_20k_select#([0-9.]+) µs/pass#µs/pass#1'
    'far_lod_section_mesh#median ([0-9.]+)#ms/section#1'
    'autosave_snapshot_and_encode_at_100k_edits#snapshot=([0-9.]+)#ms#0.001'
)

to_num() {
    awk -v s="$1" 'BEGIN {
        m = 1
        if (s ~ /[kK]$/) { m = 1000; sub(/[kK]$/, "", s) }
        else if (s ~ /[mM]$/) { m = 1000000; sub(/[mM]$/, "", s) }
        printf "%.10g", s * m
    }'
}

extract_printed() {
    local ere="$1"
    awk -v ere="$ere" '
        match($0, ere, a) { last = a[1] }
        END { if (last == "") exit 1; print last }
    '
}

# Latest `<date>: <number> <unit>` in the doc comment above `fn <name>(`.
pinned_from_src() {
    local name="$1" unit="$2"
    local loc file lnum start
    loc=$(grep -R --include='*.rs' -n -F "fn ${name}(" src | head -n 1) || return 0
    [ -n "$loc" ] || return 0
    file=${loc%%:*}
    lnum=${loc#*:}
    lnum=${lnum%%:*}
    start=$((lnum > 40 ? lnum - 40 : 1))
    sed -n "${start},$((lnum - 1))p" "$file" | awk -v unit="$unit" '
        match($0, /[0-9]{4}-[0-9]{2}-[0-9]{2}:[[:space:]]*[0-9.]+[kKmM]?[[:space:]]+/) {
            rest = substr($0, RSTART + RLENGTH)
            if (index(rest, unit) != 1) next
            after = substr(rest, length(unit) + 1, 1)
            if (after != "" && after != " " && after != "\t" && after != "(" && after != ";" && after != ",") next
            prefix = substr($0, RSTART, RLENGTH)
            sub(/^[^:]+:[[:space:]]*/, "", prefix)
            sub(/[[:space:]]+$/, "", prefix)
            pin = prefix
        }
        END { if (pin != "") print pin }
    '
}

median3() {
    printf '%s\n' "$1" "$2" "$3" | sort -n | awk 'NR == 2 { print; exit }'
}

echo "building release lib tests..." >&2
cargo test --release --no-run --lib

rows=()
json_rows=()

for spec in "${BENCHES[@]}"; do
    IFS='#' read -r name ere unit scale <<< "$spec"
    [ -n "${scale:-}" ] || scale=1

    vals=()
    skipped=
    for i in $(seq 1 "$RUNS"); do
        echo "==> $name  run $i/$RUNS" >&2
        set +e
        out=$(cargo test --release --lib "$name" -- --ignored --nocapture 2>&1)
        rc=$?
        set -e
        if printf '%s\n' "$out" | grep -q '^running 0 tests'; then
            echo "skip $name: no such test" >&2
            skipped=1
            break
        fi
        if [ "$rc" -ne 0 ]; then
            printf '%s\n' "$out" >&2
            exit "$rc"
        fi
        printed=$(printf '%s\n' "$out" | extract_printed "$ere") || {
            echo "$name: could not parse number from output:" >&2
            printf '%s\n' "$out" >&2
            exit 1
        }
        scaled=$(awk -v n="$printed" -v s="$scale" 'BEGIN { printf "%.10g", n * s }')
        vals+=("$scaled")
    done
    [ -n "$skipped" ] && continue

    value=$(median3 "${vals[0]}" "${vals[1]}" "${vals[2]}")
    pin_text=$(pinned_from_src "$name" "$unit" || true)
    if [ -n "$pin_text" ]; then
        pin_num=$(to_num "$pin_text")
        delta=$(awk -v v="$value" -v p="$pin_num" 'BEGIN {
            if (p == 0) { print "n/a"; exit }
            printf "%+.1f%%", (v - p) / p * 100
        }')
        delta_num=$(awk -v v="$value" -v p="$pin_num" 'BEGIN {
            if (p == 0) { print "null"; exit }
            printf "%.4g", (v - p) / p * 100
        }')
        pin_json=$(to_num "$pin_text")
    else
        pin_text="—"
        delta="—"
        delta_num=null
        pin_json=null
    fi
    value_fmt=$(awk -v v="$value" 'BEGIN { printf "%.6g", v }')
    rows+=("$name|$value_fmt|$unit|$pin_text|$delta")
    json_rows+=("$(awk -v name="$name" -v value="$value_fmt" -v unit="$unit" -v pinned="$pin_json" -v delta="$delta_num" 'BEGIN {
        printf "{\"name\":\"%s\",\"value\":%s,\"unit\":\"%s\"", name, value, unit
        if (pinned == "null") printf ",\"pinned\":null,\"delta_pct\":null"
        else printf ",\"pinned\":%s,\"delta_pct\":%s", pinned, delta
        printf "}\n"
    }')")
done

echo
printf '%s\n' "${rows[@]}" | awk -F'|' '
BEGIN {
    printf "%-48s %12s %-14s %12s %10s\n", "name", "value", "unit", "pinned", "delta %"
}
{
    printf "%-48s %12s %-14s %12s %10s\n", $1, $2, $3, $4, $5
}'

if [ -n "$JSON_OUT" ]; then
    mkdir -p "$(dirname "$JSON_OUT")"
    printf '%s\n' "${json_rows[@]}" > "$JSON_OUT"
    echo "wrote $JSON_OUT" >&2
fi
