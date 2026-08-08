#!/usr/bin/env bash
# Pre-submission gate. Run this before the final submission and after ANY change.
#
#   ./scripts/preflight.sh                 # check the local build
#   SNAKE_URL=https://... ./scripts/preflight.sh --live
#
# Every check here exists because something in it actually broke in production:
# a null field crashing the decision path, the model silently not driving,
# concurrent requests corrupting the native env, a 500 forfeiting a move.
set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0; FAIL=0
ok(){ echo "  PASS  $1"; PASS=$((PASS+1)); }
bad(){ echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

echo "=============================================="
echo " Battlesnake pre-submission gate"
echo "=============================================="
echo
echo "[1] test suite"
if PYTHONPATH=agent/src .venv/bin/python -m pytest agent/tests -q >/tmp/pf_tests.txt 2>&1; then
  ok "$(tail -1 /tmp/pf_tests.txt | tr -d '\n')"
else
  bad "tests failed"; tail -15 /tmp/pf_tests.txt
fi

echo
echo "[2] deployed checkpoint loads and the policy drives"
PYTHONPATH=agent/src .venv/bin/python - <<'PY' >/tmp/pf_drive.txt 2>&1
import os, sys
sys.path.insert(0,'agent/src')
os.environ.setdefault("MOVE_STRATEGY","model")
from battlesnake_ai.env.hisss_view_radius_fix import apply_view_radius_row_index_fix as f; f()
from battlesnake_ai.inference.runtime import SnakeRuntime, default_checkpoint_from_env
rt=SnakeRuntime(default_checkpoint_from_env(), device="cpu")
def snake(sid,cells,health=90,length=None):
    b=[{"x":int(x),"y":int(y)} for x,y in cells]
    return {"id":sid,"health":health,"body":b,"head":b[0],"length":length or len(cells)}
me=snake("me",((7,7),(7,6),(7,5)))
fog=[{"id":f"o{i}","health":None,"length":None,"body":[],"head":None} for i in range(3)]
p={"game":{"id":"pf"},"turn":5,"board":{"width":15,"height":15,
   "food":[{"x":9,"y":9}],"hazards":[],"snakes":[me,*fog]},"you":me}
rt.on_game_start(p)
mv=rt.decide_move(p); d=rt.last_decision()
assert mv in ("up","down","left","right"), mv
assert str(d["source"]).startswith(("model","veto","tactics")), d["source"]
print("OK", d["source"], d["ms"], "ms")
rt.close()
PY
if grep -q "^OK" /tmp/pf_drive.txt; then ok "$(cat /tmp/pf_drive.txt | grep ^OK)"; else bad "policy not driving"; cat /tmp/pf_drive.txt | tail -8; fi

echo
echo "[3] real Blackout replay frames"
if PYTHONPATH=agent/src .venv/bin/python scripts/test_blackout_api.py >/tmp/pf_replay.txt 2>&1; then
  ok "$(grep '^OK' /tmp/pf_replay.txt)"
else
  bad "replay smoke failed"; tail -8 /tmp/pf_replay.txt
fi

echo
echo "[4] hostile payloads never crash and keep the policy driving"
if PYTHONPATH=agent/src .venv/bin/python -m pytest agent/tests/test_payload_robustness.py -q >/tmp/pf_rob.txt 2>&1; then
  ok "$(tail -1 /tmp/pf_rob.txt | tr -d '\n')"
else
  bad "payload robustness failed"; tail -15 /tmp/pf_rob.txt
fi

echo
echo "[5] concurrent requests do not corrupt the native env"
PYTHONPATH=agent/src .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8123 >/tmp/pf_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 40); do curl -sf -m 2 http://127.0.0.1:8123/health >/dev/null 2>&1 && break; sleep 1; done
PYTHONPATH=agent/src .venv/bin/python - <<'PY' >/tmp/pf_conc.txt 2>&1
import json, time, urllib.request, concurrent.futures as cf
BASE="http://127.0.0.1:8123"
def payload(gid,t):
    y=3+(t%8)
    me={"id":"s0","health":90,"body":[{"x":3,"y":y},{"x":3,"y":max(0,y-1)}],"head":{"x":3,"y":y},"length":2}
    fog={"id":"s1","health":None,"length":None,"body":[],"head":None}
    return {"game":{"id":gid},"turn":t,"board":{"width":15,"height":15,
            "food":[{"x":7,"y":7}],"hazards":[],"snakes":[me,fog]},"you":me}
def one(a):
    g,t=a
    d=json.dumps(payload(g,t)).encode()
    r=urllib.request.Request(f"{BASE}/move",data=d,headers={"Content-Type":"application/json"})
    s=time.perf_counter()
    with urllib.request.urlopen(r,timeout=20) as resp:
        resp.read(); return (time.perf_counter()-s)*1000, resp.status
jobs=[(f"pf-{i%6}",i) for i in range(120)]
with cf.ThreadPoolExecutor(max_workers=12) as ex: res=list(ex.map(one,jobs))
lat=sorted(r[0] for r in res); codes={r[1] for r in res}
print("codes",codes,"p50",round(lat[len(lat)//2],1),"p99",round(lat[int(len(lat)*.99)],1))
PY
CRASH=$(grep -ciE "double free|corruption|Aborted|Segmentation" /tmp/pf_srv.log)
ALIVE=$(curl -sf -m 5 http://127.0.0.1:8123/health >/dev/null 2>&1 && echo yes || echo no)
STATS=$(curl -s -m 5 http://127.0.0.1:8123/stats 2>/dev/null)
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
if [ "$CRASH" = "0" ] && [ "$ALIVE" = "yes" ] && grep -q "codes {200}" /tmp/pf_conc.txt; then
  ok "12-way concurrency clean: $(grep codes /tmp/pf_conc.txt)"
else
  bad "concurrency problem (crash=$CRASH alive=$ALIVE)"; cat /tmp/pf_conc.txt
fi
if [ -n "$STATS" ]; then
  echo "$STATS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('        server-side: fallbacks',d['fallback_moves'],'over_budget',d['over_budget_moves'],'sources',d['move_sources'])"
fi

echo
echo "[6] no endpoint returns 5xx on malformed input"
PYTHONPATH=agent/src .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8124 >/tmp/pf_srv2.log 2>&1 &
SRV2=$!
for i in $(seq 1 40); do curl -sf -m 2 http://127.0.0.1:8124/health >/dev/null 2>&1 && break; sleep 1; done
BAD5XX=0
for body in '{}' '{"game":null,"turn":null,"board":null,"you":null}' 'not-json' '{"board":{"width":0,"height":0}}'; do
  for ep in start move end; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 10 -X POST "http://127.0.0.1:8124/$ep" \
           -H 'Content-Type: application/json' -d "$body")
    [ "$code" -ge 500 ] 2>/dev/null && BAD5XX=$((BAD5XX+1))
  done
done
kill $SRV2 2>/dev/null; wait $SRV2 2>/dev/null
[ "$BAD5XX" = "0" ] && ok "12 malformed requests, no 5xx" || bad "$BAD5XX requests returned 5xx"

if [ "${1:-}" = "--live" ]; then
  BASE="${SNAKE_URL:-https://web-production-01418.up.railway.app}"
  echo
  echo "[7] live server: $BASE"
  H=$(curl -s -m 15 "$BASE/health"); S=$(curl -s -m 15 "$BASE/stats")
  echo "$H$S" | python3 -c "
import json,sys
raw=sys.stdin.read()
i=raw.find('}{')
h=json.loads(raw[:i+1]); s=json.loads(raw[i+1:])
print('        status',h['status'],'| checkpoint',h['checkpoint'].split('/')[-1],'| patch',h['view_radius_patch'])
print('        strategy',s['strategy'],'| healthy',s['healthy'],'| fallbacks',s['fallback_moves'],'| over_budget',s['over_budget_moves'])
print('        latency',s['latency_ms'])
import sys as _s
_s.exit(0 if (h['status']=='ok' and s['fallback_moves']==0 and s['over_budget_moves']==0) else 1)
" && ok "live server healthy" || bad "live server not clean"
fi

echo
echo "=============================================="
echo " $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ] && echo " READY" || echo " NOT READY - fix the failures above"
echo "=============================================="
exit "$FAIL"
