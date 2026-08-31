#!/usr/bin/env bash
# Single entrypoint for local TWFarmBot processes.
#   ./scripts/start_all.sh           start everything
#   ./scripts/start_all.sh stop
#   ./scripts/start_all.sh restart
#   ./scripts/start_all.sh status
#   ./scripts/start_all.sh logs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# ROS (and similar) site-packages break the venv; uv commands must see a clean path.
export PYTHONPATH=""

RUN_DIR="$ROOT/data/run"
LOG_DIR="$ROOT/data/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

UV=(uv run)
if [[ -f "$ROOT/.env" ]]; then
  UV=(uv run --env-file=.env)
fi

# name | uv command | port (empty = none)
SERVICES=(
  "os|twfarmbot-os|3001"
  "api|twfarmbot-api|8000"
  "ui|twfarmbot-ui|8501"
  "worker|twfarmbot-worker|"
)

pidfile() { echo "$RUN_DIR/$1.pid"; }
logfile() { echo "$LOG_DIR/$1.log"; }

port_listening() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -q ":${port} "
  else
    netstat -an 2>/dev/null | grep -E "[.:]${port} .*LISTEN" >/dev/null
  fi
}

is_running() {
  local name="$1"
  local pf
  pf="$(pidfile "$name")"
  [[ -f "$pf" ]] || return 1
  local pid
  pid="$(cat "$pf")"
  kill -0 "$pid" 2>/dev/null
}

start_one() {
  local name="$1" cmd="$2" port="${3:-}"
  if is_running "$name"; then
    echo "$name already running (pid $(cat "$(pidfile "$name")"))"
    return
  fi
  echo "starting ${name}..."
  nohup "${UV[@]}" "$cmd" >>"$(logfile "$name")" 2>&1 &
  echo $! >"$(pidfile "$name")"
  if [[ -n "$port" ]]; then
    local i=0
    while (( i < 60 )); do
      if port_listening "$port"; then
        echo "$name listening on :$port (pid $(cat "$(pidfile "$name")"))"
        return
      fi
      if ! is_running "$name"; then
        echo "$name exited; last log lines:" >&2
        tail -n 20 "$(logfile "$name")" >&2 || true
        return 1
      fi
      sleep 0.5
      i=$((i + 1))
    done
    echo "$name still starting (pid $(cat "$(pidfile "$name")")); see $(logfile "$name")"
  else
    echo "$name started (pid $(cat "$(pidfile "$name")"))"
  fi
}

stop_one() {
  local name="$1"
  if ! is_running "$name"; then
    rm -f "$(pidfile "$name")"
    echo "$name not running"
    return
  fi
  local pid
  pid="$(cat "$(pidfile "$name")")"
  echo "stopping ${name} (pid ${pid})..."
  kill "$pid" 2>/dev/null || true
  local i=0
  while kill -0 "$pid" 2>/dev/null && (( i < 20 )); do
    sleep 0.25
    i=$((i + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$(pidfile "$name")"
}

cmd_start() {
  local row name bin port
  for row in "${SERVICES[@]}"; do
    IFS='|' read -r name bin port <<<"$row"
    start_one "$name" "$bin" "$port"
  done
  echo
  echo "OS      http://localhost:3001"
  echo "UI      http://localhost:8501"
  echo "API     http://localhost:8000/docs"
}

cmd_stop() {
  local i row name bin port
  for (( i=${#SERVICES[@]}-1; i>=0; i-- )); do
    row="${SERVICES[$i]}"
    IFS='|' read -r name bin port <<<"$row"
    stop_one "$name"
  done
}

cmd_status() {
  local row name bin port
  for row in "${SERVICES[@]}"; do
    IFS='|' read -r name bin port <<<"$row"
    if is_running "$name"; then
      echo "$name running  pid=$(cat "$(pidfile "$name")")"
    else
      echo "$name stopped"
    fi
  done
}

cmd_logs() {
  local files=() row name
  for row in "${SERVICES[@]}"; do
    IFS='|' read -r name _ <<<"$row"
    [[ -f "$(logfile "$name")" ]] && files+=("$(logfile "$name")")
  done
  if (( ${#files[@]} == 0 )); then
    echo "no log files yet under $LOG_DIR" >&2
    exit 1
  fi
  tail -F "${files[@]}"
}

usage() {
  echo "usage: $0 [start|stop|restart|status|logs]" >&2
  exit 2
}

case "${1:-start}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  -h|--help|help) usage ;;
  *) usage ;;
esac
