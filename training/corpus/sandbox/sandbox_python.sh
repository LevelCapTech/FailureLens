#!/bin/sh
# Drop-in "python" for training.corpus.build --python: runs the interpreter inside
# a locked-down container so third-party test code never executes on the host.
#
# Isolation: --network none, non-root host uid, read-only root fs, all capabilities
# dropped, no-new-privileges, cpu/memory/pids limits, a hard in-container timeout
# and a per-test timeout. Only the current directory (the private work copy), the
# PYTHONPATH directories (read-only) and the trace output directory are mounted.
# $HOME, ~/.ssh, ~/.config and tokens are never mounted or passed through.
#
# FL_SANDBOX_DRY_RUN=1 prints the docker command (one argument per line) instead
# of running it; the unit tests use this.
set -eu

IMAGE=${FL_SANDBOX_IMAGE:-failurelens-corpus-sandbox:v02}
work=$(pwd -P)
timeout_s=${FL_SANDBOX_TIMEOUT:-1500}

safe() {
  case "$1" in
    "" | *[!A-Za-z0-9_./+-]*)
      echo "sandbox: unsafe path: $1" >&2
      exit 64
      ;;
  esac
}
safe "$work"

args_file=$(mktemp)
trap 'rm -f "$args_file"' EXIT
{
  printf '%s\n' run --rm --init --network none --read-only
  printf '%s\n' --cap-drop ALL --security-opt no-new-privileges
  printf '%s\n' --user "$(id -u):$(id -g)"
  printf '%s\n' --memory "${FL_SANDBOX_MEMORY:-3g}" --memory-swap "${FL_SANDBOX_MEMORY:-3g}"
  printf '%s\n' --cpus "${FL_SANDBOX_CPUS:-2}" --pids-limit 512
  printf '%s\n' --tmpfs /tmp:rw,nosuid,nodev,size=512m
  printf '%s\n' -e HOME=/tmp -e LANG=C.UTF-8
  printf '%s\n' -e "PYTEST_ADDOPTS=--timeout=${FL_SANDBOX_TEST_TIMEOUT:-60}"
  printf '%s\n' -v "$work:$work" -w "$work"
  for name in PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONHASHSEED NO_COLOR \
    FL_TRACE_ROOT FL_TRACE_SKIP FL_TRACE_OUT FL_TRACE_CALLS; do
    eval "is_set=\${$name+x}"
    if [ -n "$is_set" ]; then
      eval "val=\${$name}"
      printf '%s\n' -e "$name=$val"
    fi
  done
  if [ -n "${PYTHONPATH:-}" ]; then
    old_ifs=$IFS
    IFS=:
    for entry in $PYTHONPATH; do
      [ -n "$entry" ] || continue
      safe "$entry"
      case "$entry" in
        "$work" | "$work"/*) ;;
        *) printf '%s\n' -v "$entry:$entry:ro" ;;
      esac
    done
    IFS=$old_ifs
  fi
  if [ -n "${FL_TRACE_OUT:-}" ]; then
    out_dir=$(dirname "$FL_TRACE_OUT")
    safe "$out_dir"
    printf '%s\n' -v "$out_dir:$out_dir"
  fi
  printf '%s\n' "$IMAGE" timeout -s KILL "$timeout_s" /opt/venv/bin/python
  for a in "$@"; do printf '%s\n' "$a"; done
} >"$args_file"

if [ "${FL_SANDBOX_DRY_RUN:-}" = "1" ]; then
  cat "$args_file"
  exit 0
fi

# Arguments are newline separated; pytest node ids and paths never contain newlines.
set --
while IFS= read -r line; do set -- "$@" "$line"; done <"$args_file"
exec docker "$@"
