# 蓝绿部署脚本执行说明

本文档说明 `blue-green-setup.sh`、`blue-green-deploy.sh`、`blue-green-switch.sh` 三个脚本的使用顺序和常用命令。

## 默认约定

| 项目 | 默认值 |
| --- | --- |
| 项目名 | `seedance-hub` |
| 部署目录 | `/usr/local/seedance-hub` |
| 二进制名称 | `main` |
| green 端口 | `3030` |
| blue 端口 | `3031` |
| 初始 active 槽位 | `green` |
| 健康检查接口 | `/health` |

蓝绿目录结构：

```text
/usr/local/seedance-hub/
├── blue/
│   ├── main
│   ├── .env        # PORT=3031
│   └── logs/
├── green/
│   ├── main
│   ├── .env        # PORT=3030
│   └── logs/
├── scripts/
│   ├── blue-green-setup.sh
│   ├── blue-green-deploy.sh
│   └── blue-green-switch.sh
├── .active_slot
└── nginx-upstream.conf
```

## 脚本用途

| 脚本 | 什么时候执行 | 作用 |
| --- | --- | --- |
| `blue-green-setup.sh` | 只在首次初始化蓝绿部署时执行一次 | 创建 `blue/green` 目录、复制当前 `main` 和 `.env`、生成 Supervisor 配置和 Nginx upstream 文件 |
| `blue-green-deploy.sh` | 每次发布新二进制时执行 | 把新二进制部署到非 active 槽位，重启该槽位，健康检查通过后切换 Nginx 流量 |
| `blue-green-switch.sh` | 需要手动切换或回滚时执行 | 在 blue/green 两个已存在槽位之间切换流量 |

## 1. 首次初始化

执行前确认服务器上已有：

```text
/usr/local/seedance-hub/main
/usr/local/seedance-hub/.env
/usr/local/seedance-hub/scripts/blue-green-setup.sh
```

执行初始化：

```bash
sudo bash /usr/local/seedance-hub/scripts/blue-green-setup.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --binary-name main \
  --blue-port 3031 \
  --green-port 3030 \
  --active-slot green
```

注意：如果 `/usr/local/seedance-hub/blue` 或 `/usr/local/seedance-hub/green` 已存在，说明已经初始化过，不要重复执行 `blue-green-setup.sh`。

加载 Supervisor 配置：

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status seedance-hub-blue seedance-hub-green
```

## 2. 配置 Nginx

在 Seedance Hub 的 Nginx `server` 块中加入：

```nginx
include /usr/local/seedance-hub/nginx-upstream.conf;
```

把固定端口代理：

```nginx
proxy_pass http://127.0.0.1:3030;
```

改为：

```nginx
proxy_pass http://127.0.0.1:$active_port;
```

测试并重载 Nginx：

```bash
sudo /usr/local/nginx/sbin/nginx -t
sudo /usr/local/nginx/sbin/nginx -s reload
```

如果旧的单实例 Supervisor 程序还在运行，可以在确认蓝绿程序正常后停止并禁用旧配置：

```bash
sudo supervisorctl stop seedance-hub
sudo mv /etc/supervisor/conf.d/seedance-hub.conf /etc/supervisor/conf.d/seedance-hub.conf.bak
sudo supervisorctl reread
sudo supervisorctl update
```

## 3. 发布新版本

假设新二进制已上传到：

```text
/usr/local/seedance-hub/server-260710_1200
```

执行：

```bash
bash /usr/local/seedance-hub/scripts/blue-green-deploy.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --binary-name main \
  --binary /usr/local/seedance-hub/server-260710_1200 \
  --blue-port 3031 \
  --green-port 3030 \
  --health-path /health \
  --health-timeout 30
```

脚本会自动完成：

```text
1. 读取当前 .active_slot
2. 找到非 active 槽位
3. 把新二进制复制到非 active 槽位
4. 重启非 active 槽位的 Supervisor 程序
5. 检查 http://localhost:<非active端口>/health
6. 健康检查通过后更新 nginx-upstream.conf
7. 执行 nginx -t
8. reload Nginx
9. 更新 .active_slot
```

如果健康检查失败，脚本会退出，流量仍保留在旧 active 槽位。

## 4. 手动切换或回滚

切到 green，也就是 `3030`：

```bash
bash /usr/local/seedance-hub/scripts/blue-green-switch.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --port 3030 \
  --blue-port 3031 \
  --green-port 3030 \
  --health-path /health
```

切到 blue，也就是 `3031`：

```bash
bash /usr/local/seedance-hub/scripts/blue-green-switch.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --port 3031 \
  --blue-port 3031 \
  --green-port 3030 \
  --health-path /health
```

紧急情况下跳过健康检查：

```bash
bash /usr/local/seedance-hub/scripts/blue-green-switch.sh \
  --project seedance-hub \
  --deploy-dir /usr/local/seedance-hub \
  --port 3030 \
  --skip-health
```

`--skip-health` 只建议在确认目标槽位可用但健康检查接口异常时使用。

## 5. 查看当前蓝绿状态

查看当前 active 槽位：

```bash
cat /usr/local/seedance-hub/.active_slot
```

查看 Nginx 当前转发端口：

```bash
cat /usr/local/seedance-hub/nginx-upstream.conf
```

查看 blue/green 的端口配置：

```bash
grep -n "PORT=" /usr/local/seedance-hub/blue/.env /usr/local/seedance-hub/green/.env
```

查看 Supervisor 状态：

```bash
sudo supervisorctl status seedance-hub-blue seedance-hub-green
```

查看端口监听：

```bash
sudo ss -lntp | grep -E '3030|3031'
```

健康检查：

```bash
curl -i http://127.0.0.1:3030/health
curl -i http://127.0.0.1:3031/health
```

判断规则：

```text
.active_slot = green，且 nginx-upstream.conf 中 active_port=3030
=> 当前 green 对外服务

.active_slot = blue，且 nginx-upstream.conf 中 active_port=3031
=> 当前 blue 对外服务
```

## 6. 常见问题

### 重复执行 setup 报错

如果看到类似：

```text
Blue-green slot directories already exist
```

说明已经初始化过。后续发布新版本只执行 `blue-green-deploy.sh`。

### deploy 健康检查失败

先查看非 active 槽位日志：

```bash
tail -n 200 /usr/local/seedance-hub/blue/logs/seedance-hub.err.log
tail -n 200 /usr/local/seedance-hub/green/logs/seedance-hub.err.log
```

再检查对应端口：

```bash
curl -i http://127.0.0.1:3030/health
curl -i http://127.0.0.1:3031/health
```

### Nginx reload 后没有切流

检查 include 是否生效：

```bash
sudo /usr/local/nginx/sbin/nginx -T | grep -n "active_port\\|proxy_pass"
cat /usr/local/seedance-hub/nginx-upstream.conf
```

确认 Nginx 中使用的是：

```nginx
proxy_pass http://127.0.0.1:$active_port;
```

而不是固定的：

```nginx
proxy_pass http://127.0.0.1:3030;
```
