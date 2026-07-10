#!/usr/bin/env bash
#
# Manually switch seedance-hub traffic between blue and green slots.
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
TARGET_PORT=3030
BLUE_PORT=3031
GREEN_PORT=3030
HEALTH_PATH="/health"
SKIP_HEALTH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT="$2"; shift 2 ;;
        --deploy-dir) DEPLOY_DIR="$2"; shift 2 ;;
        --port) TARGET_PORT="$2"; shift 2 ;;
        --blue-port) BLUE_PORT="$2"; shift 2 ;;
        --green-port) GREEN_PORT="$2"; shift 2 ;;
        --health-path) HEALTH_PATH="$2"; shift 2 ;;
        --skip-health) SKIP_HEALTH=true; shift ;;
        -h|--help)
            echo "Usage: $0 --port <port> [options]"
            echo ""
            echo "Options:"
            echo "  --project      Supervisor program prefix (default: seedance-hub)"
            echo "  --deploy-dir   Deployment root (default: /usr/local/seedance-hub)"
            echo "  --port         Target port (default: 3030)"
            echo "  --blue-port    Blue slot HTTP port (default: 3031)"
            echo "  --green-port   Green slot HTTP port (default: 3030)"
            echo "  --health-path  Health check path (default: /health)"
            echo "  --skip-health  Switch without checking target health"
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$TARGET_PORT" == "$BLUE_PORT" ]]; then
    TARGET_SLOT="blue"
elif [[ "$TARGET_PORT" == "$GREEN_PORT" ]]; then
    TARGET_SLOT="green"
else
    log_error "Target port $TARGET_PORT does not match blue($BLUE_PORT) or green($GREEN_PORT)"
    exit 1
fi

ACTIVE_SLOT_FILE="$DEPLOY_DIR/.active_slot"
CURRENT_SLOT=""
if [[ -f "$ACTIVE_SLOT_FILE" ]]; then
    CURRENT_SLOT="$(cat "$ACTIVE_SLOT_FILE")"
fi

echo ""
echo "============================================"
echo "  Blue-green switch - $PROJECT"
echo "============================================"
echo "  Current slot: ${CURRENT_SLOT:-unknown}"
echo "  Target slot:  $TARGET_SLOT ($TARGET_PORT)"
echo "============================================"
echo ""

if [[ "$CURRENT_SLOT" == "$TARGET_SLOT" ]]; then
    log_warn "Already on target slot $TARGET_SLOT"
    exit 0
fi

if [[ "$SKIP_HEALTH" == "true" ]]; then
    log_step "1/3 Skip health check"
else
    log_step "1/3 Health check http://localhost:${TARGET_PORT}${HEALTH_PATH}"
    HTTP_CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://localhost:${TARGET_PORT}${HEALTH_PATH}" 2>/dev/null || echo "000")"
    if [[ "$HTTP_CODE" != "200" ]]; then
        log_error "Target slot is not healthy (HTTP $HTTP_CODE)"
        log_error "Use --skip-health only for emergency switching"
        exit 1
    fi
    log_info "Target slot is healthy"
fi

log_step "2/3 Switch Nginx traffic"
NGINX_CONF="$DEPLOY_DIR/nginx-upstream.conf"
echo "set \$active_port $TARGET_PORT;" > "$NGINX_CONF"

NGINX_BIN="$(command -v nginx 2>/dev/null || echo "/usr/local/nginx/sbin/nginx")"
sudo "$NGINX_BIN" -t
sudo "$NGINX_BIN" -s reload
log_info "Nginx reloaded"

log_step "3/3 Update active slot marker"
echo "$TARGET_SLOT" > "$ACTIVE_SLOT_FILE"
log_info "Active slot updated to $TARGET_SLOT"

echo ""
echo "============================================"
log_info "Switch complete"
echo "  Active slot: $TARGET_SLOT ($TARGET_PORT)"
echo "============================================"
