# MySQL SQL to Go GORM Model Converter

这是一个将MySQL的CREATE TABLE语句转换为Go语言GORM模型的Python脚本。

## 功能特性

- 解析MySQL的CREATE TABLE语句，无需连接数据库
- 支持所有常用MySQL数据类型
- 智能类型映射（MySQL类型到Go类型）
- 正确处理UNSIGNED类型（使用uint32, uint64等）
- 支持NULL值处理（自动使用指针类型）
- 生成完整的GORM标签（包含列名、主键、自增、默认值、非空约束、注释等）
- 自动处理表注释和字段注释
- 支持驼峰命名转换
- 支持单表转换或从文件读取
- 支持标准输入读取

## 使用方法

### 1. 命令行直接传SQL

```bash
python sql_to_gorm.py "CREATE TABLE users (id int PRIMARY KEY, name varchar(100));"
```

### 2. 从文件读取SQL

```bash
# 创建SQL文件
echo "CREATE TABLE users (
  id int(11) unsigned NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  name varchar(100) NOT NULL COMMENT '用户名',
  created_at timestamp DEFAULT CURRENT_TIMESTAMP
);" > create_table.sql

# 转换
python sql_to_gorm.py --file create_table.sql
```

### 3. 从标准输入读取

```bash
python sql_to_gorm.py --stdin
# 然后粘贴SQL，按Ctrl+D结束
```

### 4. 保存到文件

```bash
python sql_to_gorm.py "CREATE TABLE users (...)" --output user.go
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `sql` | SQL语句（可选） |
| `--file` | 从文件读取SQL |
| `--stdin` | 从标准输入读取SQL |
| `--package` | Go包名（默认: models） |
| `--output` | 输出文件路径（默认输出到标准输出） |

## 支持的MySQL类型映射

### 整型
- `tinyint` → `int8`
- `smallint` → `int16`
- `mediumint/int` → `int32`
- `bigint` → `int64`

### 无符号整型
- `tinyint unsigned` → `uint8`
- `smallint unsigned` → `uint16`
- `int unsigned` → `uint32`
- `bigint unsigned` → `uint64`

### 浮点型
- `float` → `float32`
- `double/decimal` → `float64`

### 字符串
- `char/varchar/text系列` → `string`

### 时间日期
- `date/datetime/timestamp` → `time.Time`

### 布尔型
- `tinyint(1)` → `bool`

### 二进制
- `binary/blob系列` → `[]byte`

## 输出示例

输入：
```sql
CREATE TABLE `kk_defense_bandwidth_bill_month` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `user_id` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '会员ID',
  `periods` varchar(20) NOT NULL DEFAULT '' COMMENT '账期(格式: YYYY-MM)',
  `scheme_type` tinyint(2) unsigned NOT NULL DEFAULT '0' COMMENT '计费项目: 1-带宽 2-IP',
  `money_total` decimal(10,2) unsigned NOT NULL DEFAULT '0.00' COMMENT '费用',
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='防护带宽月度账单明细表';
```

输出：
```go
package models

import (
    "time"
)

// DefenseBandwidthBillMonth 防护带宽月度账单明细表
type DefenseBandwidthBillMonth struct {
    // 自增ID
    Id                   uint32          `json:"id"`                    `gorm:"column:id;AUTO_INCREMENT;NOT NULL;comment:'自增ID'"`

    // 会员ID
    userId               uint32          `json:"user_id"`               `gorm:"column:user_id;NOT NULL;default:0;comment:'会员ID'"`

    // 账期(格式: YYYY-MM)
    periods              string          `json:"periods"`               `gorm:"column:periods;NOT NULL;default:;comment:'账期(格式: YYYY-MM)'"`

    // 计费项目: 1-带宽 2-IP
    schemeType           uint8           `json:"scheme_type"`           `gorm:"column:scheme_type;NOT NULL;default:0;comment:'计费项目: 1-带宽 2-IP'"`

    // 费用
    moneyTotal           float64         `json:"money_total"`           `gorm:"column:money_total;NOT NULL;default:0.00;type:decimal(10,2) unsigned;comment:'费用'"`

    // 创建时间
    createTime           time.Time       `json:"create_time"`           `gorm:"column:create_time;NOT NULL;default:CURRENT_TIMESTAMP;type:timestamp;comment:'创建时间'"`
}

func (d *DefenseBandwidthBillMonth) TableName() string {
    return "kk_defense_bandwidth_bill_month"
}
```

## 注意事项

1. 确保SQL语句格式正确，特别是反引号的使用
2. 脚本会自动识别主键（通过PRIMARY KEY定义）
3. 可NULL字段会自动转换为指针类型（除string和[]byte外）
4. 表名会自动转换为大驼峰命名作为结构体名
5. 字段名会自动从下划线命名转换为驼峰命名
6. 特殊处理tinyint(1)为bool类型

## 简化使用

创建一个批处理脚本批量转换：

```bash
#!/bin/bash
# 批量转换SQL文件为Go模型

for sql_file in *.sql; do
    go_file="${sql_file%.sql}.go"
    python sql_to_gorm.py --file "$sql_file" --output "$go_file"
    echo "转换完成: $sql_file → $go_file"
done
```



  1. 多表解析 - 可以在一个文件中解析多个CREATE TABLE语句
  2. 数据库名前缀支持 - 支持database.table格式
  3. 批量输出到目录 - 使用--output-dir参数为每个表生成单独的文件
  4. 智能单表/多表模式 - 自动检测表数量并选择合适的输出方式

  使用示例

  # 解析单个表（原有功能保持不变）
  python sql_to_gorm.py "CREATE TABLE users (...)"

  # 解析文件中的多个表，输出到标准输出
  python sql_to_gorm.py --file tables.sql

  # 解析文件中的多个表，每个表生成一个文件
  python sql_to_gorm.py --file tables.sql --output-dir ./models

  # 强制使用多表模式
  python sql_to_gorm.py "CREATE TABLE ..." --multiple

  # 从标准输入读取多个表
  python sql_to_gorm.py --stdin --output-dir ./models

  支持的SQL格式

  - 标准CREATE TABLE语句
  - 带数据库名前缀的表名（如db_name.table_name）
  - 多个CREATE TABLE语句在同一文件中
  - 每个表可以有自己的ENGINE、CHARSET等选项
  - 支持注释和分隔线


## 许可证

MIT License