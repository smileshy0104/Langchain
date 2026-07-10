# 新项目蓝绿部署 Runbook

这份文档用于后续把新的 Go 服务接入当前这套部署方式：

- GitHub Actions 构建二进制
- self-hosted GitHub runner 执行部署
- Supervisor 管理 blue / green 两个进程
- Nginx 通过 `nginx-upstream.conf` 切换流量
- `/health` 健康检查通过后再切流

下文用占位符表示新项目参数：

| 占位符 | 含义 | 示例 |
|---|---|---|
| `<project>` | 项目名、Supervisor 程序名前缀 | `seedance-hub` |
| `<deploy_dir>` | 服务器部署目录 | `/usr/local/seedance-hub` |
| `<binary_name>` | 编译后的二进制名 | `main` |
| `<domain>` | Nginx server_name | `seedance-hub.dev.wepcc.com` |
| `<green_port>` | green 槽端口，通常沿用旧端口 | `3030` |
| `<blue_port>` | blue 槽端口，新开一个端口 | `3031` |
| `<runner_user>` | GitHub runner 用户 | `github-runner` |
| `<db_name>` | MySQL 数据库名 | `seedance_hub` |
| `<db_user>` | MySQL 应用账号 | `seedance_rw` |
| `<db_port>` | 宿主机 MySQL 端口 | `6303` |
| `<db_socket>` | 宿主机 MySQL socket 路径 | `/data/mysql/seedance-hub/run/mysqld.sock` |
| `<redis_port>` | Redis 监听端口 | `9736` |

## 1. 项目代码要求

### 健康检查接口

每个接入蓝绿部署的项目都必须有一个无需登录、无需数据库强依赖的健康检查接口：

```text
GET /health
```

建议返回 HTTP 200，例如：

```json
{
  "code": 200,
  "message": "ok",
  "data": {
    "status": "ok"
  }
}
```

部署脚本只判断 HTTP 状态码是否为 `200`。如果 `/health` 依赖数据库、Redis 或第三方服务，外部依赖抖动会导致新版本无法切流，所以基础健康检查应尽量轻量。

### 构建产物

GitHub Actions 最终需要生成 Linux amd64 二进制：

```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
  -ldflags "-s -w -X 'github.com/QuantumNous/new-api/common.Version=$VERSION'" \
  -o bin/<binary_name> .
```

如果新项目不是当前 Go module，需要同步调整 `-X` 的版本变量路径。

如果后端通过 `go:embed` 打包前端 dist，workflow 中必须先构建前端，再执行 `go build`。

## 2. 全新服务器初始化

以下流程按 Ubuntu / Debian 系服务器编写。其他发行版可以替换对应包管理命令。

当前已验证的服务器示例：

```text
Debian GNU/Linux 13 (trixie)
```

Debian 13 的默认软件源通常不提供 Oracle MySQL 的 `mysql-server` 包，会提示使用 `mariadb-server` 替代。因此 Debian 13 推荐直接安装 MariaDB，应用仍通过 MySQL 协议连接。

### 基础软件

```bash
sudo apt update
sudo apt install -y \
  ca-certificates \
  curl \
  git \
  unzip \
  build-essential \
  supervisor \
  redis-server
```

数据库二选一。

Debian 13 trixie 推荐：

```bash
sudo apt install -y mariadb-server
```

Ubuntu 或已配置 Oracle MySQL 源的服务器可以尝试：

```bash
sudo apt install -y mysql-server
```

如果出现 `Package 'mysql-server' has no installation candidate`，说明当前系统源不提供 Oracle MySQL 包，通常会提示 `mariadb-server` 替代。此时安装 MariaDB：

```bash
sudo apt install -y mariadb-server
```

MariaDB 使用 MySQL 协议，应用侧 DSN 仍然使用 `tcp(127.0.0.1:3306)`。如果业务明确要求 Oracle MySQL 8，而不是 MariaDB，需要先添加 MySQL 官方 APT Repository，再安装 `mysql-server`。

如果希望用 Docker 部署 Oracle MySQL，而不是系统 MariaDB，跳到“Docker 部署 MySQL”章节。

如果服务器使用自编译 Nginx，确认二进制路径：

```bash
test -x /usr/local/nginx/sbin/nginx && /usr/local/nginx/sbin/nginx -v
```

如果使用系统 Nginx：

```bash
sudo apt install -y nginx
nginx -v
```

本文档默认 Nginx 命令路径为：

```text
/usr/local/nginx/sbin/nginx
```

如果你的服务器使用 `/usr/sbin/nginx`，需要同步调整 sudoers 和脚本里的 Nginx 命令。

### 创建运行用户和目录

建议让应用由独立用户运行，不要直接用 root：

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin <project> || true
sudo mkdir -p <deploy_dir>
sudo chown -R <project>:<project> <deploy_dir>
```

如果 GitHub runner 和应用运行用户是同一个用户，可以把 `<project>` 替换成 `<runner_user>`。如果二者不同，后续需要给 `<runner_user>` 部署目录写权限。

### 安装 Go 和 Bun

如果构建动作在 GitHub runner 所在服务器执行，需要安装 Go。也可以只在 runner 机器安装，线上运行机器不一定需要 Go。

```bash
go version
```

如未安装，按当前项目要求安装 Go 1.22+。

安装 Bun：

```bash
curl -fsSL https://bun.sh/install | bash
```

重新登录 shell 后确认：

```bash
bun --version
```

如果 GitHub Actions 使用 `oven-sh/setup-bun`，runner 执行环境会在 workflow 中安装 Bun，服务器手工部署时才需要提前安装。

### Docker 部署 MySQL

Debian 13 上如果希望使用 Oracle MySQL，推荐用 Docker 跑官方 MySQL 镜像。先检查 Docker 是否已经安装：

```bash
docker --version || true
systemctl status docker --no-pager || true
```

如果未安装 Docker，按 Docker 官方 Debian apt repository 方式安装：

```bash
sudo apt update
sudo apt install -y ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/debian
Suites: $(. /etc/os-release && echo "$VERSION_CODENAME")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

如果希望 `<runner_user>` 可以执行 Docker 命令：

```bash
sudo usermod -aG docker <runner_user>
```

重新登录 `<runner_user>` 的 shell 后生效。生产服务器上也可以继续使用 `sudo docker ...`，权限更收敛。

先查看现有 Docker 数据库，避免误停已有业务：

```bash
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Ports}}' | grep 3306 || true
```

当前 seedance-hub 服务器已观察到：

```text
kk-ai-mysql   mysql:8   127.0.0.1:3306->3306/tcp
```

不要修改或停止这个容器。新项目推荐单独起一个 MySQL 容器，并绑定到另一个本机端口，例如 `127.0.0.1:6303`。这不是公网暴露，只有服务器本机可以连接。

如果要求宿主机完全不监听 MySQL TCP 端口，可以使用后面的“Socket 方案”。

#### 推荐方案：绑定本机端口

创建目录：

```bash
sudo mkdir -p /opt/mysql-<project> /data/mysql/<project>
sudo chmod 700 /opt/mysql-<project>
cd /opt/mysql-<project>
```

生成密码：

```bash
openssl rand -base64 32
openssl rand -base64 32
```

写入 `/opt/mysql-<project>/.env`：

```env
MYSQL_ROOT_PASSWORD=<root_password>
MYSQL_DATABASE=<db_name>
MYSQL_USER=<db_user>
MYSQL_PASSWORD=<db_password>
MYSQL_PORT=<db_port>
```

保护 `.env`：

```bash
sudo chmod 600 /opt/mysql-<project>/.env
```

写入 `/opt/mysql-<project>/compose.yml`：

```yaml
services:
  mysql:
    image: mysql:8.4
    container_name: mysql-<project>
    restart: unless-stopped
    ports:
      - "127.0.0.1:${MYSQL_PORT:-6303}:3306"
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ${MYSQL_DATABASE}
      MYSQL_USER: ${MYSQL_USER}
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}
      TZ: Asia/Shanghai
    command:
      - --character-set-server=utf8mb4
      - --collation-server=utf8mb4_unicode_ci
      - --default-time-zone=+08:00
    volumes:
      - /data/mysql/<project>:/var/lib/mysql
    healthcheck:
      test: ["CMD-SHELL", "mysqladmin ping -h 127.0.0.1 -uroot -p$${MYSQL_ROOT_PASSWORD} --silent"]
      interval: 10s
      timeout: 5s
      retries: 10
```

启动：

```bash
cd /opt/mysql-<project>
sudo docker compose --env-file .env -f compose.yml up -d
sudo docker compose --env-file .env -f compose.yml ps
sudo docker compose --env-file .env -f compose.yml logs -f mysql
```

验证：

```bash
mysql -h 127.0.0.1 -P <db_port> -u <db_user> -p <db_name> -e "SELECT 1;"
```

如果宿主机没有 `mysql` 客户端，也可以进入容器验证：

```bash
sudo docker exec -it mysql-<project> mysql -u root -p
```

应用 `.env` 中的 DSN 模板：

```env
SQL_DSN=<db_user>:<url_encoded_db_password>@tcp(127.0.0.1:<db_port>)/<db_name>?parseTime=true
```

注意：

- `ports` 必须绑定 `127.0.0.1`，不要绑定 `0.0.0.0`，否则数据库会暴露到公网网卡。
- 如果宿主机 3306 已经被 `kk-ai-mysql` 占用，新项目使用 `<db_port>=6303`。
- 如果密码包含 `@`、`#`、`:`、`/` 等特殊字符，写入 DSN 时必须 URL encode。
- `mysql:8.4` 是 MySQL 官方镜像标签；如果业务依赖旧版行为，可以改成 `mysql:8.0`。

备份示例：

```bash
cd /opt/mysql-<project>
set -a
. ./.env
set +a
sudo docker exec mysql-<project> mysqldump -u root -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE" > "<project>-$(date +%F).sql"
```

#### 可选方案：完全不监听端口

如果明确要求宿主机完全不监听 MySQL TCP 端口，可以去掉 `ports:`，通过 Unix Socket 连接。

目录：

```bash
sudo mkdir -p /opt/mysql-<project> /data/mysql/<project>/data /data/mysql/<project>/run
sudo chmod 700 /opt/mysql-<project>
sudo chmod 755 /data/mysql/<project>/run
cd /opt/mysql-<project>
```

`compose.yml` 中不要写 `ports:`，并增加 socket 配置：

```yaml
services:
  mysql:
    image: mysql:8.4
    container_name: mysql-<project>
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ${MYSQL_DATABASE}
      MYSQL_USER: ${MYSQL_USER}
      MYSQL_PASSWORD: ${MYSQL_PASSWORD}
      TZ: Asia/Shanghai
    command:
      - --character-set-server=utf8mb4
      - --collation-server=utf8mb4_unicode_ci
      - --default-time-zone=+08:00
      - --socket=/var/run/mysqld/mysqld.sock
    volumes:
      - /data/mysql/<project>/data:/var/lib/mysql
      - /data/mysql/<project>/run:/var/run/mysqld
```

验证：

```bash
mysql --protocol=socket -S <db_socket> -u <db_user> -p <db_name> -e "SELECT 1;"
```

应用 `.env` 中的 DSN 模板：

```env
SQL_DSN=<db_user>:<url_encoded_db_password>@unix(<db_socket>)/<db_name>?parseTime=true
```

### MySQL / MariaDB 初始化

确认数据库服务名：

```bash
if systemctl list-unit-files | grep -q '^mysql.service'; then
  DB_SERVICE=mysql
else
  DB_SERVICE=mariadb
fi
echo "$DB_SERVICE"
```

启动数据库：

```bash
sudo systemctl enable "$DB_SERVICE"
sudo systemctl start "$DB_SERVICE"
sudo systemctl status "$DB_SERVICE" --no-pager
```

首次加固：

```bash
sudo mysql_secure_installation
```

创建数据库和应用账号。密码不要写进文档或仓库，先在服务器上生成：

```bash
openssl rand -base64 32
```

进入 MySQL：

```bash
sudo mysql
```

执行：

```sql
CREATE DATABASE `<db_name>` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER '<db_user>'@'127.0.0.1' IDENTIFIED BY '<db_password>';
GRANT ALL PRIVILEGES ON `<db_name>`.* TO '<db_user>'@'127.0.0.1';
FLUSH PRIVILEGES;
```

验证：

```bash
mysql -h 127.0.0.1 -u <db_user> -p <db_name> -e "SELECT 1;"
```

应用 `.env` 中的 DSN 模板：

```env
SQL_DSN=<db_user>:<url_encoded_db_password>@tcp(127.0.0.1:<db_port>)/<db_name>?parseTime=true
SQL_MAX_IDLE_CONNS=100
SQL_MAX_OPEN_CONNS=1000
SQL_MAX_LIFETIME=60
```

如果密码中包含 `@`、`#`、`:`、`/` 等特殊字符，需要 URL encode 后再写入 DSN。

### Redis 初始化

复制一份项目专用 Redis 配置，避免直接改系统默认实例：

```bash
sudo cp /etc/redis/redis.conf /etc/redis/redis-<project>.conf
```

编辑：

```bash
sudo vim /etc/redis/redis-<project>.conf
```

建议设置：

```conf
bind 127.0.0.1 ::1
port <redis_port>
protected-mode yes
requirepass <redis_password>
supervised systemd
dir /var/lib/redis-<project>
logfile /var/log/redis/redis-<project>.log
dbfilename dump-<project>.rdb
appendonly yes
appendfilename "appendonly-<project>.aof"
```

创建数据目录：

```bash
sudo mkdir -p /var/lib/redis-<project>
sudo chown redis:redis /var/lib/redis-<project>
sudo chmod 750 /var/lib/redis-<project>
```

创建 systemd 服务：

```bash
sudo cp /lib/systemd/system/redis-server.service /etc/systemd/system/redis-<project>.service
sudo vim /etc/systemd/system/redis-<project>.service
```

把 `ExecStart` 改成：

```ini
ExecStart=/usr/bin/redis-server /etc/redis/redis-<project>.conf --supervised systemd --daemonize no
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable redis-<project>
sudo systemctl start redis-<project>
sudo systemctl status redis-<project> --no-pager
```

验证：

```bash
redis-cli -h 127.0.0.1 -p <redis_port> -a '<redis_password>' ping
```

应用 `.env` 中的连接模板：

```env
REDIS_CONN_STRING=redis://default:<url_encoded_redis_password>@127.0.0.1:<redis_port>
REDIS_POOL_SIZE=20
MEMORY_CACHE_ENABLED=true
```

Redis 密码同样需要 URL encode。

### 防火墙和监听范围

MySQL / MariaDB 和 Redis 建议只监听本机：

```bash
ss -lntp | grep -E '3306|<redis_port>|<green_port>|<blue_port>'
```

应看到 MySQL / MariaDB / Redis 为 `127.0.0.1`，应用端口可以是 `0.0.0.0` 或 `127.0.0.1`，但实际对外入口应由 Nginx 负责。

如果使用 UFW，只开放 HTTP/HTTPS 和必要 SSH：

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status
```

不要把 MySQL 和 Redis 端口暴露到公网。

## 3. 应用环境文件

创建 `<deploy_dir>/.env`：

```bash
sudo vim <deploy_dir>/.env
sudo chown <project>:<project> <deploy_dir>/.env
sudo chmod 600 <deploy_dir>/.env
```

基础模板：

```env
HOST=0.0.0.0
PORT=<green_port>

VERSION=1.0.0

SQL_DSN=<db_user>:<url_encoded_db_password>@tcp(127.0.0.1:<db_port>)/<db_name>?parseTime=true
SQL_MAX_IDLE_CONNS=100
SQL_MAX_OPEN_CONNS=1000
SQL_MAX_LIFETIME=60

REDIS_CONN_STRING=redis://default:<url_encoded_redis_password>@127.0.0.1:<redis_port>
MEMORY_CACHE_ENABLED=true
REDIS_POOL_SIZE=20

NODE_TYPE=master

GLOBAL_API_RATE_LIMIT_ENABLE=false
GLOBAL_WEB_RATE_LIMIT_ENABLE=false
CRITICAL_RATE_LIMIT_ENABLE=false

ERROR_LOG_ENABLED=true
```

确认应用能读取 `.env` 后，再进入蓝绿初始化。

## 4. 服务器初始目录

先准备旧版或首次可运行版本：

```bash
sudo mkdir -p <deploy_dir>
sudo cp <binary_name> <deploy_dir>/<binary_name>
sudo cp .env <deploy_dir>/.env
sudo chmod +x <deploy_dir>/<binary_name>
```

`.env` 至少需要包含服务监听地址和端口：

```env
HOST=0.0.0.0
PORT=<green_port>
```

不要把生产 `.env`、数据库密码、Redis 密码写进仓库或文档。

## 5. 初始化蓝绿目录

在服务器上执行：

```bash
sudo bash scripts/blue-green-setup.sh \
  --project <project> \
  --deploy-dir <deploy_dir> \
  --binary-name <binary_name> \
  --green-port <green_port> \
  --blue-port <blue_port> \
  --active-slot green \
  --user <runner_user>
```

初始化后目录结构应类似：

```text
<deploy_dir>/
├── blue/
│   ├── <binary_name>
│   ├── .env        # PORT=<blue_port>
│   └── logs/
├── green/
│   ├── <binary_name>
│   ├── .env        # PORT=<green_port>
│   └── logs/
├── .active_slot
└── nginx-upstream.conf
```

默认让 `green` 作为初始 active slot，原因是它通常沿用旧服务端口，可以降低首次迁移风险。

## 6. Supervisor 配置

`blue-green-setup.sh` 会生成：

```text
/etc/supervisor/conf.d/<project>-bluegreen.conf
```

加载配置：

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status <project>-green <project>-blue
```

如果原来有单进程配置，例如：

```text
/etc/supervisor/conf.d/<project>.conf
```

确认 blue / green 正常后，停止并备份旧配置：

```bash
sudo supervisorctl stop <project>
sudo mv /etc/supervisor/conf.d/<project>.conf /etc/supervisor/conf.d/<project>.conf.bak
sudo supervisorctl reread
sudo supervisorctl update
```

## 7. Nginx 配置

在对应 vhost 的 `server` 块中加入：

```nginx
include <deploy_dir>/nginx-upstream.conf;
```

把固定端口：

```nginx
proxy_pass http://127.0.0.1:<green_port>;
```

改成：

```nginx
proxy_pass http://127.0.0.1:$active_port;
```

保留原来的 proxy header、timeout、streaming 相关配置。

检查并 reload：

```bash
sudo /usr/local/nginx/sbin/nginx -t
sudo /usr/local/nginx/sbin/nginx -s reload
```

## 8. GitHub runner 权限

runner 用户需要能写入部署目录中的 blue / green 槽位、active marker 和 Nginx upstream include：

```bash
sudo chown -R <runner_user>:<runner_user> <deploy_dir>/blue <deploy_dir>/green
sudo chown <runner_user>:<runner_user> <deploy_dir>/.active_slot <deploy_dir>/nginx-upstream.conf
```

同时通过 `visudo` 增加最小 sudo 权限：

```bash
sudo visudo -f /etc/sudoers.d/<runner_user>
```

写入：

```text
<runner_user> ALL=(ALL) NOPASSWD: /usr/bin/supervisorctl restart <project>-blue
<runner_user> ALL=(ALL) NOPASSWD: /usr/bin/supervisorctl restart <project>-green
<runner_user> ALL=(ALL) NOPASSWD: /usr/local/nginx/sbin/nginx -t
<runner_user> ALL=(ALL) NOPASSWD: /usr/local/nginx/sbin/nginx -s reload
```

验证：

```bash
sudo -u <runner_user> sudo /usr/bin/supervisorctl restart <project>-blue
sudo -u <runner_user> sudo /usr/bin/supervisorctl restart <project>-green
sudo -u <runner_user> sudo /usr/local/nginx/sbin/nginx -t
sudo -u <runner_user> sudo /usr/local/nginx/sbin/nginx -s reload
```

## 9. GitHub Actions 配置

dev 环境推荐：

```yaml
name: Deploy to Dev

on:
  push:
    branches:
      - dev
  workflow_dispatch:

permissions:
  contents: read

env:
  PROJECT_NAME: <project>
  DEPLOY_DIR: <deploy_dir>
  BLUE_PORT: "<blue_port>"
  GREEN_PORT: "<green_port>"
  HEALTH_PATH: /health
  HEALTH_TIMEOUT: "30"

jobs:
  build-and-deploy:
    runs-on: [self-hosted, dev]
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: oven-sh/setup-bun@v2
        with:
          bun-version: latest

      - name: Build frontend
        run: |
          set -euo pipefail
          VERSION="$(git describe --tags --always --dirty 2>/dev/null || echo dev-${GITHUB_SHA::7})"
          echo "$VERSION" > VERSION
          cd web
          bun install --frozen-lockfile
          cd default
          VITE_REACT_APP_VERSION="$VERSION" bun run build

      - name: Build backend
        run: |
          set -euo pipefail
          VERSION="$(cat VERSION)"
          go mod download
          mkdir -p bin
          CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
            -ldflags "-s -w" \
            -o bin/main .

      - name: Blue-green deploy
        run: |
          set -euo pipefail
          scripts/blue-green-deploy.sh \
            --project "$PROJECT_NAME" \
            --deploy-dir "$DEPLOY_DIR" \
            --binary bin/main \
            --binary-name main \
            --blue-port "$BLUE_PORT" \
            --green-port "$GREEN_PORT" \
            --health-path "$HEALTH_PATH" \
            --health-timeout "$HEALTH_TIMEOUT"
```

如果项目没有前端，删除 `Setup Bun` 和 `Build frontend` 步骤即可。

如果部署目录需要通过 secret 覆盖，可加：

```yaml
env:
  DEPLOY_DIR_OVERRIDE: ${{ secrets.<PROJECT>_DEPLOY_PATH }}
```

然后脚本中使用：

```bash
TARGET_DEPLOY_DIR="${DEPLOY_DIR_OVERRIDE:-$DEPLOY_DIR}"
```

## 10. 发布流程

### 首次上线

1. 在服务器准备 `<deploy_dir>/<binary_name>` 和 `<deploy_dir>/.env`。
2. 执行 `blue-green-setup.sh`。
3. 修改 Nginx 为 `$active_port`。
4. `supervisorctl reread && supervisorctl update`。
5. 检查两个 slot：

```bash
curl -s http://localhost:<green_port>/health
curl -s http://localhost:<blue_port>/health
sudo supervisorctl status <project>-green <project>-blue
```

6. push 到触发分支或手动运行 workflow。

### 日常发布

1. 合并代码到触发分支。
2. GitHub Actions 构建二进制。
3. 部署脚本写入 inactive slot。
4. 重启 inactive slot。
5. `/health` 返回 200 后切 Nginx。
6. `.active_slot` 更新为新 slot。

### 回滚

查看当前 active slot：

```bash
cat <deploy_dir>/.active_slot
cat <deploy_dir>/nginx-upstream.conf
```

切到 green：

```bash
bash scripts/blue-green-switch.sh \
  --project <project> \
  --deploy-dir <deploy_dir> \
  --port <green_port>
```

切到 blue：

```bash
bash scripts/blue-green-switch.sh \
  --project <project> \
  --deploy-dir <deploy_dir> \
  --port <blue_port>
```

紧急情况下跳过健康检查：

```bash
bash scripts/blue-green-switch.sh \
  --project <project> \
  --deploy-dir <deploy_dir> \
  --port <green_port> \
  --skip-health
```

## 11. 上线检查清单

服务器：

```bash
ls -la <deploy_dir>
ls -la <deploy_dir>/blue <deploy_dir>/green
cat <deploy_dir>/.active_slot
cat <deploy_dir>/nginx-upstream.conf
sudo supervisorctl status <project>-green <project>-blue
sudo /usr/local/nginx/sbin/nginx -t
```

MySQL / MariaDB / Redis：

```bash
systemctl status mysql --no-pager || systemctl status mariadb --no-pager
systemctl status redis-<project> --no-pager

# TCP 方案
mysql -h 127.0.0.1 -P <db_port> -u <db_user> -p <db_name> -e "SELECT 1;"

# Socket 方案
mysql --protocol=socket -S <db_socket> -u <db_user> -p <db_name> -e "SELECT 1;"

redis-cli -h 127.0.0.1 -p <redis_port> -a '<redis_password>' ping
```

健康检查：

```bash
curl -i http://localhost:<green_port>/health
curl -i http://localhost:<blue_port>/health
curl -i http://<domain>/health
```

权限：

```bash
sudo -u <runner_user> test -w <deploy_dir>/blue
sudo -u <runner_user> test -w <deploy_dir>/green
sudo -u <runner_user> test -w <deploy_dir>/.active_slot
sudo -u <runner_user> test -w <deploy_dir>/nginx-upstream.conf
```

Action：

```bash
git status --short
bash -n scripts/blue-green-setup.sh scripts/blue-green-deploy.sh scripts/blue-green-switch.sh
```

## 12. 常见问题

### `sudo: a password is required`

说明 sudoers 没有允许 runner 执行对应命令。补充：

```text
<runner_user> ALL=(ALL) NOPASSWD: /usr/bin/supervisorctl restart <project>-blue
<runner_user> ALL=(ALL) NOPASSWD: /usr/bin/supervisorctl restart <project>-green
<runner_user> ALL=(ALL) NOPASSWD: /usr/local/nginx/sbin/nginx -t
<runner_user> ALL=(ALL) NOPASSWD: /usr/local/nginx/sbin/nginx -s reload
```

### `Health check failed`

检查 inactive slot：

```bash
sudo supervisorctl status <project>-blue <project>-green
tail -n 100 <deploy_dir>/blue/logs/<project>.err.log
tail -n 100 <deploy_dir>/green/logs/<project>.err.log
curl -i http://localhost:<blue_port>/health
curl -i http://localhost:<green_port>/health
```

常见原因：

- `.env` 中端口错误。
- 新二进制没有执行权限。
- Supervisor 进程没启动。
- `/health` 路由未注册。
- 服务启动依赖数据库或 Redis，但服务器环境变量不正确。

### `Package 'mysql-server' has no installation candidate`

当前系统源没有 Oracle MySQL 包。Debian 13 trixie 默认就是这种情况。继续部署有两个选择：

```bash
# 方案 A：使用系统源提供的 MariaDB，最快
sudo apt install -y mariadb-server
sudo systemctl enable mariadb
sudo systemctl start mariadb
sudo systemctl status mariadb --no-pager
```

MariaDB 使用 MySQL 协议，后续建库、建用户、应用 DSN 基本不变。

如果必须安装 Oracle MySQL 8，需要添加 MySQL 官方 APT Repository 后再安装 `mysql-server`，不要继续反复执行原来的 apt 命令。

### MySQL / MariaDB 连接失败

检查：

```bash
systemctl status mysql --no-pager || systemctl status mariadb --no-pager
ss -lntp | grep 3306
mysql -h 127.0.0.1 -u <db_user> -p <db_name> -e "SELECT 1;"
```

常见原因：

- `<db_user>` 只授权了 `localhost`，但 DSN 使用 `127.0.0.1`。
- 密码包含特殊字符，但 DSN 没有 URL encode。
- 数据库名或权限不正确。
- MySQL 未启动或只监听了其他地址。

### MariaDB 启动失败：3306 已被占用

如果看到类似日志：

```text
Can't start server: Bind on TCP/IP port: Address already in use
Do you already have another server running on port: 3306 ?
```

先确认谁占用了端口：

```bash
sudo ss -lntp | grep ':3306'
sudo lsof -iTCP:3306 -sTCP:LISTEN -P -n || true
ps -ef | grep -E 'mysql|maria|mysqld' | grep -v grep
```

如果显示 `docker-proxy`，说明某个 Docker 容器把 3306 映射到了宿主机：

```bash
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Ports}}' | grep 3306
docker ps -a --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Ports}}' | grep 3306
```

这种情况下优先确认容器里的数据库是否就是要复用的实例：

```bash
mysql -h 127.0.0.1 -P 3306 -uroot -p -e "SELECT VERSION();"
```

如果确认复用 Docker 中的数据库，就不要启动宿主机 `mariadb.service`，避免端口冲突。后续直接在该实例里建库和建用户。

如果 3306 上已经有 MySQL / MariaDB 正常运行，不要再启动新的 `mariadb.service`，直接复用现有实例：

```bash
mysql -h 127.0.0.1 -P 3306 -uroot -p -e "SELECT VERSION();"
```

然后在现有实例里创建 `<db_name>` 和 `<db_user>`。

如果 3306 被异常残留进程占用，先确认它不是其他业务正在使用的数据库，再处理：

```bash
sudo systemctl status mysql --no-pager || true
sudo systemctl status mariadb --no-pager || true
```

如果确认要使用 MariaDB 默认服务，可以停止冲突服务后再启动：

```bash
sudo systemctl stop mysql || true
sudo systemctl start mariadb
sudo systemctl status mariadb --no-pager
```

如果必须保留 3306 上已有数据库，又想额外启动一个 MariaDB 实例，需要给新实例改端口和独立数据目录；这种方式会增加维护复杂度，优先复用现有数据库实例。

### Redis 连接失败

检查：

```bash
systemctl status redis-<project> --no-pager
ss -lntp | grep <redis_port>
redis-cli -h 127.0.0.1 -p <redis_port> -a '<redis_password>' ping
```

常见原因：

- Redis 密码和 `.env` 不一致。
- Redis 密码包含特殊字符，但连接串没有 URL encode。
- Redis 实例未启动。
- 配置文件仍使用默认 `port 6379`。

### Nginx reload 后没有切流

检查：

```bash
cat <deploy_dir>/nginx-upstream.conf
sudo /usr/local/nginx/sbin/nginx -T | grep -n "active_port\\|proxy_pass"
```

确认 vhost 中已经 include：

```nginx
include <deploy_dir>/nginx-upstream.conf;
proxy_pass http://127.0.0.1:$active_port;
```

### 前端依赖构建失败

如果 monorepo 中存在多个前端，注意依赖版本可能被 workspace 顶层提升。例如 classic 前端的 `date-fns-tz@1.x` 需要兼容 `date-fns@2.x`，但新版前端可能依赖 `date-fns@4.x`。

处理原则：

- 对旧前端显式声明它需要的兼容版本。
- 更新 lockfile。
- 必要时在 Rsbuild/Vite 配置中加 alias，避免解析到另一个 workspace 的版本。
- CI 中使用 `bun install --frozen-lockfile`，保证 lockfile 和 package 声明一致。

本地验证：

```bash
cd web
bun install --frozen-lockfile
cd classic
bun run build
```

### 生成的 `bin/main` 不要提交

`bin/main` 是构建产物。提交前确认：

```bash
git status --short
```

只提交源码、脚本、workflow、lockfile，不提交本地生成的二进制。

## 13. Seedance Hub 示例参数

| 参数 | 值 |
|---|---|
| `<project>` | `seedance-hub` |
| `<deploy_dir>` | `/usr/local/seedance-hub` |
| `<binary_name>` | `main` |
| `<domain>` | `seedance-hub.dev.wepcc.com` |
| `<green_port>` | `3030` |
| `<blue_port>` | `3031` |
| `<runner_user>` | `github-runner` |
| `<db_name>` | `seedance_hub` |
| `<redis_port>` | `9736` |

对应 sudoers：

```text
github-runner ALL=(ALL) NOPASSWD: /usr/bin/supervisorctl restart seedance-hub-blue
github-runner ALL=(ALL) NOPASSWD: /usr/bin/supervisorctl restart seedance-hub-green
github-runner ALL=(ALL) NOPASSWD: /usr/local/nginx/sbin/nginx -t
github-runner ALL=(ALL) NOPASSWD: /usr/local/nginx/sbin/nginx -s reload
```
