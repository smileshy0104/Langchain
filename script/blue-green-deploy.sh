#!/usr/bin/env bash
#
# Deploy a new seedance-hub binary to the inactive blue-green slot.
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }
log_step() { echo -e "${BLUE}[STEP]${NC}  $(date '+%H:%M:%S') $*"; }

PROJECT="seedance-hub"
DEPLOY_DIR="/usr/local/seedance-hub"
BINARY_NAME="main"
BINARY_SRC=""
BLUE_PORT=3031
GREEN_PORT=3030
HEALTH_PATH="/health"
HEALTH_TIMEOUT=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT="$2"; shift 2 ;;
        --deploy-dir) DEPLOY_DIR="$2"; shift 2 ;;
        --binary-name) BINARY_NAME="$2"; shift 2 ;;
        --binary) BINARY_SRC="$2"; shift 2 ;;
        --blue-port) BLUE_PORT="$2"; shift 2 ;;
        --green-port) GREEN_PORT="$2"; shift 2 ;;
        --health-path) HEALTH_PATH="$2"; shift 2 ;;
        --health-timeout) HEALTH_TIMEOUT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --binary <path> [options]"
            echo ""
            echo "Options:"
            echo "  --project        Supervisor program prefix (default: seedance-hub)"
            echo "  --deploy-dir     Deployment root (default: /usr/local/seedance-hub)"
            echo "  --binary-name    Binary filename in each slot (default: main)"
            echo "  --binary         New binary path (required)"
            echo "  --blue-port      Blue slot HTTP port (default: 3031)"
            echo "  --green-port     Green slot HTTP port (default: 3030)"
            echo "  --health-path    Health check path (default: /health)"
            echo "  --health-timeout Health check timeout seconds (default: 30)"
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$BINARY_SRC" ]]; then
    log_error "--binary is required"
    exit 1
fi

if [[ ! -f "$BINARY_SRC" ]]; then
    log_error "Binary does not exist: $BINARY_SRC"
    exit 1
fi

ACTIVE_SLOT_FILE="$DEPLOY_DIR/.active_slot"
if [[ -f "$ACTIVE_SLOT_FILE" ]]; then
    ACTIVE_SLOT="$(cat "$ACTIVE_SLOT_FILE")"
else
    ACTIVE_SLOT="green"
    log_warn ".active_slot not found; assuming green is active"
fi

if [[ "$ACTIVE_SLOT" == "blue" ]]; then
    INACTIVE_SLOT="green"
    INACTIVE_PORT="$GREEN_PORT"
    ACTIVE_PORT="$BLUE_PORT"
elif [[ "$ACTIVE_SLOT" == "green" ]]; then
    INACTIVE_SLOT="blue"
    INACTIVE_PORT="$BLUE_PORT"
    ACTIVE_PORT="$GREEN_PORT"
else
    log_error ".active_slot has invalid value: $ACTIVE_SLOT"
    exit 1
fi

INACTIVE_DIR="$DEPLOY_DIR/$INACTIVE_SLOT"
if [[ ! -d "$INACTIVE_DIR" ]]; then
    log_error "Inactive slot directory does not exist: $INACTIVE_DIR"
    log_error "Run blue-green-setup.sh first"
    exit 1
fi

echo ""
echo "============================================"
echo "  Blue-green deploy - $PROJECT"
echo "============================================"
echo "  Active slot: $ACTIVE_SLOT ($ACTIVE_PORT)"
echo "  Target slot: $INACTIVE_SLOT ($INACTIVE_PORT)"
echo "  Target dir:  $INACTIVE_DIR"
echo "============================================"
echo ""

log_step "1/4 Deploy binary to $INACTIVE_SLOT"
cp "$BINARY_SRC" "$INACTIVE_DIR/${BINARY_NAME}.new"
chmod +x "$INACTIVE_DIR/${BINARY_NAME}.new"
mv -f "$INACTIVE_DIR/${BINARY_NAME}.new" "$INACTIVE_DIR/$BINARY_NAME"
log_info "Binary deployed"

log_step "2/4 Restart Supervisor program"
SUPERVISOR_PROGRAM="${PROJECT}-${INACTIVE_SLOT}"
sudo supervisorctl restart "$SUPERVISOR_PROGRAM"
log_info "Restarted $SUPERVISOR_PROGRAM"

log_step "3/4 Health check http://localhost:${INACTIVE_PORT}${HEALTH_PATH}"
HEALTH_OK=false
ELAPSED=0
INTERVAL=2
while [[ "$ELAPSED" -lt "$HEALTH_TIMEOUT" ]]; do
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
    HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:${INACTIVE_PORT}${HEALTH_PATH}" 2>/dev/null || echo "000")"
    if [[ "$HTTP_CODE" == "200" ]]; then
        HEALTH_OK=true
        log_info "Health check passed (HTTP $HTTP_CODE, ${ELAPSED}s)"
        break
    fi
    log_warn "Waiting for healthy response (HTTP $HTTP_CODE, ${ELAPSED}/${HEALTH_TIMEOUT}s)"
done

if [[ "$HEALTH_OK" != "true" ]]; then
    log_error "Health check failed for $INACTIVE_SLOT on port $INACTIVE_PORT"
    log_error "Traffic remains on $ACTIVE_SLOT ($ACTIVE_PORT)"
    exit 1
fi

log_step "4/4 Switch Nginx traffic"
NGINX_CONF="$DEPLOY_DIR/nginx-upstream.conf"
echo "set \$active_port $INACTIVE_PORT;" > "$NGINX_CONF"

NGINX_BIN="$(command -v nginx 2>/dev/null || echo "/usr/local/nginx/sbin/nginx")"
sudo "$NGINX_BIN" -t
sudo "$NGINX_BIN" -s reload

echo "$INACTIVE_SLOT" > "$ACTIVE_SLOT_FILE"
log_info "Active slot updated to $INACTIVE_SLOT"

echo ""
echo "============================================"
log_info "Blue-green deploy complete"
echo "  New active slot: $INACTIVE_SLOT ($INACTIVE_PORT)"
echo "  Previous slot:   $ACTIVE_SLOT ($ACTIVE_PORT)"
echo "============================================"
