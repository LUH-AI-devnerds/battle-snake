#!/usr/bin/env bash
# Live competition monitor. Run during a leaderboard session:
#
#   ./scripts/watch_live.sh              # refresh every 20s
#   ./scripts/watch_live.sh 10           # refresh every 10s
#
# The number that matters is `fallbacks`. Anything above 0 means the RL policy
# stopped driving and the crude heuristic answered instead -- that is a defect,
# not a slow patch, and it is how two silent outages went unnoticed.
set -uo pipefail

BASE="${SNAKE_URL:-https://web-production-01418.up.railway.app}"
INTERVAL="${1:-20}"

render() {
python3 <<'PY'
import json, sys, urllib.request, os

base = os.environ["BASE"]
try:
    with urllib.request.urlopen(base + "/stats?recent_games=8", timeout=10) as r:
        d = json.load(r)
except Exception as exc:
    print("  UNREACHABLE: %r" % (exc,))
    raise SystemExit(0)

fb = d["fallback_moves"]
ob = d["over_budget_moves"]
health = "HEALTHY" if d["healthy"] else "DEGRADED"

print("  {}   strategy={}   uptime={:.0f}s".format(health, d["strategy"], d["uptime_s"]))
print("  checkpoint {}".format(d["checkpoint"]))
print()
print("  games {:<5} moves {:<7} fallbacks {:<5} over-budget {}".format(
    d["games"], d["moves"], fb, ob))
if fb:
    print("  !! {} moves served by the last-resort heuristic "
          "- the policy was not playing".format(fb))
if ob:
    print("  !! {} moves exceeded the 500ms budget".format(ob))

l = d["latency_ms"]
print("  latency  p50 {}ms   p95 {}ms   p99 {}ms   max {}ms   (budget {:.0f}ms)".format(
    l["p50"], l["p95"], l["p99"], l["max"], l["budget"]))
print("  sources  {}".format(d["move_sources"]))

o = d["outcomes"]
total = sum(o.values())
if total:
    print("  outcomes {}   win rate {:.0f}% over {} finished games".format(
        o, 100.0 * o.get("won", 0) / total, total))
print()
print("  recent games")
for g in d["recent_games"]:
    warn = "  <-- FALLBACKS" if g["fallbacks"] else ""
    print("    {:12s} {:11s} turns={:<5} len={:<4} p95={}ms{}".format(
        g["game_id"][:10], g["outcome"], g["turns"], g["our_length"],
        g["latency_ms"]["p95"], warn))
PY
}

export BASE
while true; do
  clear
  echo "-- the sea snake -- $(date '+%H:%M:%S') -- ${BASE}"
  echo
  render
  echo
  echo "  (ctrl-c to stop; refreshing every ${INTERVAL}s)"
  sleep "${INTERVAL}"
done
