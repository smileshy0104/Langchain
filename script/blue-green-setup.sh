#!/usr/bin/env bash
#
# Initialize a blue-green deployment layout for seedance-hub.
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
BLUE_PORT=3031
GREEN_PORT=3030
ACTIVE_SLOT="green"
RUN_USER=""
SUPERVISOR_CONF=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT="$2"; shift 2 ;;
        --deploy-dir) DEPLOY_DIR="$2"; shift 2 ;;
        --binary-name) BINARY_NAME="$2"; shift 2 ;;
        --blue-port) BLUE_PORT="$2"; shift 2 ;;
        --green-port) GREEN_PORT="$2"; shift 2 ;;
        --active-slot) ACTIVE_SLOT="$2"; shift 2 ;;
        --user) RUN_USER="$2"; shift 2 ;;
        --supervisor-conf) SUPERVISOR_CONF="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: sudo $0 [options]"
            echo ""
            echo "Options:"
            echo "  --project      Supervisor program prefix (default: seedance-hub)"
            echo "  --deploy-dir   Deployment root (default: /usr/local/seedance-hub)"
            echo "  --binary-name  Binary filename (default: main)"
            echo "  --blue-port    Blue slot HTTP port (default: 3031)"
            echo "  --green-port   Green slot HTTP port (default: 3030)"
            echo "  --active-slot  Initial active slot: blue or green (default: green)"
            echo "  --user         Supervisor run user (optional)"
            echo "  --supervisor-conf Supervisor config path (default: /etc/supervisor/conf.d/<project>-bluegreen.conf)"
            exit 0
            ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$SUPERVISOR_CONF" ]]; then
    SUPERVISOR_CONF="/etc/supervisor/conf.d/${PROJECT}-bluegreen.conf"
fi

if [[ "$ACTIVE_SLOT" != "blue" && "$ACTIVE_SLOT" != "green" ]]; then
    log_error "--active-slot must be blue or green"
    exit 1
fi

if [[ -d "$DEPLOY_DIR/blue" || -d "$DEPLOY_DIR/green" ]]; then
    log_error "Blue-green slot directories already exist under $DEPLOY_DIR"
    log_error "Refusing to overwrite existing deployment state"
    exit 1
fi

if [[ ! -d "$DEPLOY_DIR" ]]; then
    log_error "Deploy directory does not exist: $DEPLOY_DIR"
    exit 1
fi

if [[ ! -f "$DEPLOY_DIR/$BINARY_NAME" ]]; then
    log_warn "Current binary not found: $DEPLOY_DIR/$BINARY_NAME"
    log_warn "Slot directories will be created, but binaries must be deployed before startup"
fi

if [[ ! -f "$DEPLOY_DIR/.env" ]]; then
    log_error "Current .env not found: $DEPLOY_DIR/.env"
    exit 1
fi

if [[ -z "$RUN_USER" ]]; then
    if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
        RUN_USER="$SUDO_USER"
    else
        RUN_USER="$(stat -c '%U' "$DEPLOY_DIR" 2>/dev/null || echo root)"
    fi
fi
RUN_GROUP=""
if id "$RUN_USER" >/dev/null 2>&1; then
    RUN_GROUP="$(id -gn "$RUN_USER")"
fi

echo ""
echo "============================================"
echo "  Blue-green setup - $PROJECT"
echo "============================================"
echo "  Deploy directory: $DEPLOY_DIR"
echo "  Binary name:      $BINARY_NAME"
echo "  Blue port:        $BLUE_PORT"
echo "  Green port:       $GREEN_PORT"
echo "  Initial active:   $ACTIVE_SLOT"
echo "  Run user:         $RUN_USER"
if [[ -n "$RUN_GROUP" ]]; then
    echo "  Run group:        $RUN_GROUP"
fi
echo "============================================"
echo ""

set_env_port() {
    local env_file="$1"
    local port="$2"

    awk -v port="$port" '
        BEGIN { found = 0 }
        /^PORT=/ {
            print "PORT=" port
            found = 1
            next
        }
        { print }
        END {
            if (found == 0) {
                print "PORT=" port
            }
        }
    ' "$env_file" > "${env_file}.tmp"
    mv "${env_file}.tmp" "$env_file"
}

log_step "1/6 Create slot directories"
for slot in blue green; do
    mkdir -p "$DEPLOY_DIR/$slot/logs"
done
log_info "Created blue and green slots"

log_step "2/6 Copy current deployment into slots"
for slot in blue green; do
    if [[ -f "$DEPLOY_DIR/$BINARY_NAME" ]]; then
        cp "$DEPLOY_DIR/$BINARY_NAME" "$DEPLOY_DIR/$slot/$BINARY_NAME"
        chmod +x "$DEPLOY_DIR/$slot/$BINARY_NAME"
    fi
    cp "$DEPLOY_DIR/.env" "$DEPLOY_DIR/$slot/.env"
done
set_env_port "$DEPLOY_DIR/blue/.env" "$BLUE_PORT"
set_env_port "$DEPLOY_DIR/green/.env" "$GREEN_PORT"
log_info "Copied binary and generated slot .env files"

log_step "3/6 Write Supervisor configuration"
USER_LINE=""
if [[ -n "$RUN_USER" && "$RUN_USER" != "root" ]]; then
    USER_LINE="user=${RUN_USER}"
fi

cat > "$SUPERVISOR_CONF" << EOF
[program:${PROJECT}-blue]
command=${DEPLOY_DIR}/blue/${BINARY_NAME}
directory=${DEPLOY_DIR}/blue
autostart=true
autorestart=true
startsecs=5
startretries=3
${USER_LINE}
stderr_logfile=${DEPLOY_DIR}/blue/logs/${PROJECT}.err.log
stdout_logfile=${DEPLOY_DIR}/blue/logs/${PROJECT}.out.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=3

[program:${PROJECT}-green]
command=${DEPLOY_DIR}/green/${BINARY_NAME}
directory=${DEPLOY_DIR}/green
autostart=true
autorestart=true
startsecs=5
startretries=3
${USER_LINE}
stderr_logfile=${DEPLOY_DIR}/green/logs/${PROJECT}.err.log
stdout_logfile=${DEPLOY_DIR}/green/logs/${PROJECT}.out.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
stderr_logfile_maxbytes=50MB
stderr_logfile_backups=3
EOF
log_info "Supervisor config written: $SUPERVISOR_CONF"

log_step "4/6 Write Nginx upstream include"
if [[ "$ACTIVE_SLOT" == "blue" ]]; then
    ACTIVE_PORT="$BLUE_PORT"
else
    ACTIVE_PORT="$GREEN_PORT"
fi
echo "set \$active_port $ACTIVE_PORT;" > "$DEPLOY_DIR/nginx-upstream.conf"
log_info "Nginx upstream initialized to $ACTIVE_SLOT ($ACTIVE_PORT)"

log_step "5/6 Write active slot marker"
echo "$ACTIVE_SLOT" > "$DEPLOY_DIR/.active_slot"
log_info "Active slot marker written"

log_step "6/6 Set ownership"
if id "$RUN_USER" >/dev/null 2>&1; then
    chown -R "$RUN_USER:$RUN_GROUP" "$DEPLOY_DIR/blue" "$DEPLOY_DIR/green" "$DEPLOY_DIR/.active_slot" "$DEPLOY_DIR/nginx-upstream.conf"
    log_info "Ownership set to $RUN_USER:$RUN_GROUP"
else
    log_warn "User $RUN_USER does not exist; skipped chown"
fi

echo ""
echo "============================================"
log_info "Blue-green setup complete"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Update Nginx to include: ${DEPLOY_DIR}/nginx-upstream.conf"
echo "  2. Change proxy_pass to: http://127.0.0.1:\$active_port"
echo "  3. Stop old Supervisor program: sudo supervisorctl stop ${PROJECT}"
echo "  4. Load new programs: sudo supervisorctl reread && sudo supervisorctl update"
echo "  5. Reload Nginx: sudo nginx -t && sudo nginx -s reload"
