#!/usr/bin/env bash
# Run one OpenROAD-flow-scripts design with the Docker toolchain.
#
# Host `make` fails on this machine because YOSYS_EXE points at
# tools/install, which was never built outside the image. GCD completed
# with flow/util/docker_shell. This script does the same for any design
# that has designs/<platform>/<design>/config.mk.
#
# Usage, from Ubuntu:
#   ./scripts/run_orfs_design.sh aes
#   ./scripts/run_orfs_design.sh ibex
#   ./scripts/run_orfs_design.sh jpeg
#
# Overrides: ORFS_ROOT, OR_IMAGE, PLATFORM, FLOW_VARIANT, NUM_CORES.

set -euo pipefail

PLATFORM="${PLATFORM:-sky130hd}"
FLOW_VARIANT="${FLOW_VARIANT:-base}"
OR_IMAGE="${OR_IMAGE:-openroad/orfs:latest}"
ORFS_ROOT="${ORFS_ROOT:-/home/administrator/OpenROAD-flow-scripts}"

# Paths inside openroad/orfs, not on the host.
CONTAINER_YOSYS="/OpenROAD-flow-scripts/tools/install/yosys/bin/yosys"
CONTAINER_OPENROAD="/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
CONTAINER_KLAYOUT="/usr/bin/klayout"

usage() {
  cat <<EOF
Usage: $(basename "$0") <design>

Run the sky130hd base RTL-to-GDS flow for <design> inside ${OR_IMAGE}.
The design name selects flow/designs/<platform>/<design>/config.mk.

Environment:
  ORFS_ROOT       OpenROAD-flow-scripts checkout (default: ${ORFS_ROOT})
  OR_IMAGE        Docker image (default: ${OR_IMAGE})
  PLATFORM        Platform directory (default: ${PLATFORM})
  FLOW_VARIANT    Flow variant (default: ${FLOW_VARIANT})
  NUM_CORES       OpenROAD threads (default: chosen from free memory)
EOF
}

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'FAILURE %s\n' "${DESIGN:-}" >&2
  printf '%s\n' "$*" >&2
  exit 1
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 2
fi

DESIGN="$1"
shift
if [[ $# -ne 0 ]]; then
  usage >&2
  die "Unexpected argument: $1"
fi

if [[ ! "$DESIGN" =~ ^[A-Za-z][A-Za-z0-9_-]*$ ]]; then
  die "Design name '${DESIGN}' is not a design directory name."
fi

if ! command -v docker >/dev/null 2>&1; then
  die "docker is not on PATH. Run this script inside Ubuntu, where the ORFS image is available."
fi

if ! docker image inspect "$OR_IMAGE" >/dev/null 2>&1; then
  die "Docker image '${OR_IMAGE}' is not available. Pull it before running the flow."
fi

FLOW_DIR="${ORFS_ROOT}/flow"
CONFIG_REL="designs/${PLATFORM}/${DESIGN}/config.mk"
CONFIG="${FLOW_DIR}/${CONFIG_REL}"
DOCKER_SHELL="${FLOW_DIR}/util/docker_shell"

if [[ ! -d "$FLOW_DIR" ]]; then
  die "ORFS flow directory not found: ${FLOW_DIR}"
fi
if [[ ! -x "$DOCKER_SHELL" ]]; then
  die "docker_shell is missing or not executable: ${DOCKER_SHELL}"
fi
if [[ ! -f "$CONFIG" ]]; then
  die "Design config not found: ${CONFIG}"
fi

make_assignment() {
  local key="$1"
  local file="$2"
  local value
  value="$(sed -n -E "s/^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*[?]?=[ \t]*//p" "$file" | tail -1 || true)"
  value="${value%%#*}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

NICKNAME="$(make_assignment DESIGN_NICKNAME "$CONFIG")"
VARIANT="$(make_assignment FLOW_VARIANT "$CONFIG")"
CONFIG_PLATFORM="$(make_assignment PLATFORM "$CONFIG")"

if [[ -z "$NICKNAME" || "$NICKNAME" == *'$'* ]]; then
  NICKNAME="$DESIGN"
fi
if [[ -n "$VARIANT" && "$VARIANT" != *'$'* ]]; then
  FLOW_VARIANT="$VARIANT"
fi
if [[ -n "$CONFIG_PLATFORM" && "$CONFIG_PLATFORM" != *'$'* ]]; then
  PLATFORM="$CONFIG_PLATFORM"
fi

RESULTS_DIR="${FLOW_DIR}/results/${PLATFORM}/${NICKNAME}/${FLOW_VARIANT}"
REPORTS_DIR="${FLOW_DIR}/reports/${PLATFORM}/${NICKNAME}/${FLOW_VARIANT}"

REQUIRED=(
  "${RESULTS_DIR}/6_final.v"
  "${RESULTS_DIR}/6_final.sdc"
  "${RESULTS_DIR}/6_final.spef"
  "${REPORTS_DIR}/6_finish.rpt"
)

if [[ -z "${NUM_CORES:-}" ]]; then
  mem_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  cpus="$(nproc 2>/dev/null || echo 1)"
  by_mem=$((mem_kb / 2000000))
  if [[ "$by_mem" -lt 1 ]]; then
    by_mem=1
  fi
  if [[ "$by_mem" -gt "$cpus" ]]; then
    by_mem="$cpus"
  fi
  NUM_CORES="$by_mem"
fi

log "Design: ${DESIGN}"
log "Config: ${CONFIG}"
log "Image: ${OR_IMAGE}"
log "Threads: ${NUM_CORES}"
log "Results: ${RESULTS_DIR}"

# A root container leaves results the host user cannot update. Reclaim only
# this design's directories, using the image as root, before docker_shell
# runs make as the host uid.
reclaim_tree() {
  local dir="$1"
  if [[ ! -e "$dir" || -w "$dir" ]]; then
    return 0
  fi
  local rel="${dir#"${FLOW_DIR}/"}"
  log "Reclaiming ${dir}"
  docker run --rm -v "${FLOW_DIR}:/work" "$OR_IMAGE" \
    chown -R "$(id -u):$(id -g)" "/work/${rel}"
}

reclaim_tree "${FLOW_DIR}/results/${PLATFORM}/${NICKNAME}"
reclaim_tree "${FLOW_DIR}/logs/${PLATFORM}/${NICKNAME}"
reclaim_tree "${FLOW_DIR}/reports/${PLATFORM}/${NICKNAME}"
reclaim_tree "${FLOW_DIR}/objects/${PLATFORM}/${NICKNAME}"

cd "$FLOW_DIR"

# docker_shell forwards these names into the container. A host value such as
# the missing tools/install/yosys path must not win.
unset YOSYS_EXE OPENROAD_EXE OPENSTA_EXE KLAYOUT_CMD
export YOSYS_EXE="$CONTAINER_YOSYS"
export OPENROAD_EXE="$CONTAINER_OPENROAD"
export KLAYOUT_CMD="$CONTAINER_KLAYOUT"
export OR_IMAGE

set +e
./util/docker_shell make "DESIGN_CONFIG=./${CONFIG_REL}" "NUM_CORES=${NUM_CORES}"
flow_status=$?
set -e

missing=()
for path in "${REQUIRED[@]}"; do
  if [[ ! -s "$path" ]]; then
    missing+=("$path")
  fi
done

if [[ "$flow_status" -ne 0 || ${#missing[@]} -gt 0 ]]; then
  printf 'FAILURE %s\n' "$DESIGN" >&2
  if [[ "$flow_status" -ne 0 ]]; then
    printf 'ORFS exited with status %s\n' "$flow_status" >&2
  fi
  if [[ ${#missing[@]} -gt 0 ]]; then
    printf 'Missing or empty finish artifacts:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
  fi
  exit 1
fi

printf 'SUCCESS %s\n' "$DESIGN"
printf '6_final.v    %s\n' "${RESULTS_DIR}/6_final.v"
printf '6_final.sdc  %s\n' "${RESULTS_DIR}/6_final.sdc"
printf '6_final.spef %s\n' "${RESULTS_DIR}/6_final.spef"
printf '6_finish.rpt %s\n' "${REPORTS_DIR}/6_finish.rpt"
