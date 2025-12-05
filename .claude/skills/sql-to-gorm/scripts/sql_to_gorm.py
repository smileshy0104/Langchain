#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MySQL SQL to Go GORM Model Converter
将MySQL的CREATE TABLE语句转换为Go语言的GORM模型

用法:
    python sql_to_gorm.py "CREATE TABLE users (id int PRIMARY KEY, name varchar(100));"
    python sql_to_gorm.py --file create_table.sql
    python sql_to_gorm.py --stdin
"""

import re
import json
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import textwrap


@dataclass
class ColumnInfo:
    """列信息"""
    name: str
    go_field_name: str
    go_type: str
    json_tag: str
    gorm_tag: str
    comment: str
    is_primary_key: bool = False
    is_auto_increment: bool = False
    is_nullable: bool = True
    default_value: str = None


class MySQLSQLParser:
    """MySQL SQL解析器"""

    # MySQL类型到Go类型的映射
    TYPE_MAPPING = {
        # 整型
        'tinyint': 'int8',
        'smallint': 'int16',
        'mediumint': 'int32',
        'int': 'int32',
        'integer': 'int32',
        'bigint': 'int64',

        # 无符号整型
        'int unsigned': 'uint32',
        'int(11) unsigned': 'uint32',
        'tinyint unsigned': 'uint8',
        'smallint unsigned': 'uint16',
        'mediumint unsigned': 'uint32',
        'bigint unsigned': 'uint64',
        'tinyint(2) unsigned': 'uint8',
        'tinyint(1) unsigned': 'bool',  # tinyint(1) unsigned 通常用于布尔值
        'decimal(10,2) unsigned': 'float64',
        'decimal unsigned': 'float64',

        # 浮点型
        'float': 'float32',
        'double': 'float64',
        'decimal': 'float64',
        'decimal(10,2)': 'float64',
        'decimal(10,2) unsigned': 'float64',

        # 字符串
        'char': 'string',
        'varchar': 'string',
        'tinytext': 'string',
        'text': 'string',
        'mediumtext': 'string',
        'longtext': 'string',

        # 时间日期
        'date': 'time.Time',
        'datetime': 'time.Time',
        'timestamp': 'time.Time',
        'time': 'time.Time',
        'year': 'int32',

        # 布尔型
        'boolean': 'bool',
        'bool': 'bool',
        'tinyint(1)': 'bool',

        # 二进制
        'binary': '[]byte',
        'varbinary': '[]byte',
        'tinyblob': '[]byte',
        'blob': '[]byte',
        'mediumblob': '[]byte',
        'longblob': '[]byte',

        # JSON
        'json': 'string',
    }

    def __init__(self):
        self.table_name = None
        self.table_comment = None
        self.columns = []

    def parse_sql(self, sql: str) -> bool:
        """解析SQL语句"""
        # 清理SQL
        sql = self._clean_sql(sql)

        # 提取表名（支持数据库名.表名格式）
        table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`?\w+`?\.)?`?(\w+)`?\s*\(', sql, re.IGNORECASE)
        if not table_match:
            print("✗ 无法找到表名")
            return False

        self.table_name = table_match.group(1)

        # 提取表注释
        table_comment_match = re.search(r'COMMENT\s*=\s*[\'"]([^\'\"]+)[\'"]', sql, re.IGNORECASE)
        if table_comment_match:
            self.table_comment = table_comment_match.group(1)

        # 提取列定义部分
        # 找到第一个(和最后一个)之间的内容
        columns_section_start = sql.find('(')
        columns_section_end = sql.rfind(')')

        if columns_section_start == -1 or columns_section_end == -1 or columns_section_end <= columns_section_start:
            print("✗ 无法找到列定义部分")
            return False

        columns_section = sql[columns_section_start + 1:columns_section_end]

        # 移除外部的约束定义（如PRIMARY KEY, KEY等）
        columns_section = re.sub(r',\s*(?:PRIMARY\s+KEY|UNIQUE\s+KEY|KEY|INDEX|CONSTRAINT|FOREIGN\s+KEY)\s+.*$', '',
                                columns_section, flags=re.IGNORECASE | re.MULTILINE)

        # 分割列定义
        column_definitions = []
        current_col = ""
        paren_count = 0

        for char in columns_section:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                if current_col.strip():
                    column_definitions.append(current_col.strip())
                current_col = ""
                continue

            current_col += char

        if current_col.strip():
            column_definitions.append(current_col.strip())

        # 解析每个列定义
        primary_keys = []

        # 首先提取PRIMARY KEY定义
        for col_def in column_definitions:
            if re.match(r'^PRIMARY\s+KEY', col_def, re.IGNORECASE):
                # 提取主键列名
                pk_match = re.search(r'PRIMARY\s+KEY\s*\(\s*`?(\w+)`?\s*\)', col_def, re.IGNORECASE)
                if pk_match:
                    primary_keys.append(pk_match.group(1))

        # 解析列
        for col_def in column_definitions:
            if re.match(r'^PRIMARY\s+KEY|^UNIQUE\s+KEY|^KEY|^INDEX|^CONSTRAINT|^FOREIGN\s+KEY', col_def, re.IGNORECASE):
                continue

            column_info = self._parse_column_definition(col_def)
            if column_info:
                # 检查是否是主键
                if column_info.name in primary_keys:
                    column_info.is_primary_key = True
                self.columns.append(column_info)

        return True

    def _clean_sql(self, sql: str) -> str:
        """清理SQL语句"""
        # 移除注释
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

        # 合并多行
        sql = ' '.join(sql.split())

        return sql

    def _parse_column_definition(self, col_def: str) -> Optional[ColumnInfo]:
        """解析单个列定义"""
        # 基本格式: `name` type [NOT NULL | NULL] [DEFAULT value] [AUTO_INCREMENT] [COMMENT 'comment']

        # 提取列名和类型（支持unsigned）
        # 匹配格式：`name` type [UNSIGNED] [其他选项]
        name_type_match = re.match(r'^`?(\w+)`?\s+([a-zA-Z]+(?:\([^)]+\))?(?:\s+unsigned)?)', col_def, re.IGNORECASE)
        if not name_type_match:
            return None

        column_name = name_type_match.group(1)
        column_type_full = name_type_match.group(2).lower()

        
        # 提取注释
        comment_match = re.search(r"COMMENT\s+'([^']*)'", col_def, re.IGNORECASE)
        comment = comment_match.group(1) if comment_match else ''

        # 判断是否可为NULL
        is_nullable = True
        if re.search(r'\bNOT\s+NULL\b', col_def, re.IGNORECASE):
            is_nullable = False

        # 提取默认值
        default_value = None
        default_match = re.search(r'DEFAULT\s+(\S+)', col_def, re.IGNORECASE)
        if default_match:
            default_value = default_match.group(1).strip("'\"")

        # 判断是否自增
        is_auto_increment = bool(re.search(r'AUTO_INCREMENT', col_def, re.IGNORECASE))

        # 判断是否是主键（在列定义中）
        is_primary_key = bool(re.search(r'\bPRIMARY\s+KEY\b', col_def, re.IGNORECASE))

        # 转换字段名为Go命名规范（驼峰命名）
        go_field_name = self._to_camel_case(column_name)

        # 特殊处理：如果字段名是id，改为Id
        if column_name.lower() == 'id':
            go_field_name = 'Id'

        # 确定Go类型
        go_type = self._get_go_type(column_type_full, is_nullable)

        # 生成JSON标签
        json_tag = f'`json:"{column_name}"`'

        # 生成GORM标签
        gorm_tag = self._generate_gorm_tag(
            column_name, column_type_full, is_nullable,
            default_value, is_primary_key, is_auto_increment, comment
        )

        return ColumnInfo(
            name=column_name,
            go_field_name=go_field_name,
            go_type=go_type,
            json_tag=json_tag,
            gorm_tag=gorm_tag,
            comment=comment,
            is_primary_key=is_primary_key,
            is_auto_increment=is_auto_increment,
            is_nullable=is_nullable,
            default_value=default_value
        )

    def _to_camel_case(self, snake_str: str) -> str:
        """将下划线命名转换为驼峰命名"""
        components = snake_str.split('_')
        # 第一个单词首字母小写，其余首字母大写
        return components[0] + ''.join(x.capitalize() for x in components[1:])

    def _get_go_type(self, mysql_type: str, is_nullable: bool) -> str:
        """获取Go类型"""
        # 特殊处理tinyint(1)作为bool（包括unsigned版本）
        if mysql_type.startswith('tinyint(1)'):
            return 'bool'

        # 特殊处理无符号类型
        if 'unsigned' in mysql_type:
            # 无符号整型
            if 'int' in mysql_type:
                if 'bigint' in mysql_type:
                    base_type = 'uint64'
                elif 'smallint' in mysql_type:
                    base_type = 'uint16'
                elif 'mediumint' in mysql_type:
                    base_type = 'uint32'
                elif 'tinyint' in mysql_type:
                    base_type = 'uint8'
                else:  # int
                    base_type = 'uint32'
                # 如果可为空，使用指针类型
                if is_nullable:
                    base_type = f'*{base_type}'
                return base_type
            # 无符号decimal保持float64
            elif 'decimal' in mysql_type:
                return 'float64'

        # 查找类型映射
        base_type = None
        for mysql_pattern, go_type in self.TYPE_MAPPING.items():
            # 先检查精确匹配
            if mysql_type == mysql_pattern:
                base_type = go_type
                break
            # 再检查前缀匹配
            elif mysql_type.startswith(mysql_pattern):
                base_type = go_type
                break

        if not base_type:
            # 默认使用string
            base_type = 'string'

        # 如果可为空，使用指针类型
        if is_nullable and base_type not in ['string', '[]byte']:
            if base_type == 'time.Time':
                base_type = '*time.Time'
            else:
                base_type = f'*{base_type}'

        return base_type

    def _generate_gorm_tag(self, column_name: str, mysql_type: str,
                          is_nullable: bool, default_value: str,
                          is_primary_key: bool, is_auto_increment: bool,
                          comment: str) -> str:
        """生成GORM标签"""
        gorm_parts = []

        # 列名
        gorm_parts.append(f'column:{column_name}')

        # 主键
        if is_primary_key:
            gorm_parts.append('primary_key')

        # 自增
        if is_auto_increment:
            gorm_parts.append('AUTO_INCREMENT')

        # 非空约束
        if not is_nullable:
            gorm_parts.append('NOT NULL')

        # 默认值
        if default_value is not None:
            if default_value.upper() == 'CURRENT_TIMESTAMP':
                gorm_parts.append("default:CURRENT_TIMESTAMP")
            else:
                gorm_parts.append(f"default:{default_value}")

        # 特殊类型
        if 'decimal' in mysql_type.lower():
            # 保持原有的decimal定义
            gorm_parts.append(f"type:{mysql_type}")
        elif 'timestamp' in mysql_type.lower() or 'datetime' in mysql_type.lower():
            gorm_parts.append('type:timestamp')

        # 注释
        gorm_parts.append(f"comment:'{comment}'")

        return f'`gorm:"{";".join(gorm_parts)}"`'

    def generate_go_model(self, package_name: str = 'models') -> str:
        """生成Go模型代码"""
        # 表名转换为结构体名（大驼峰）
        struct_name = self._table_name_to_struct_name(self.table_name)

        lines = []
        lines.append(f'package {package_name}')
        lines.append('')
        lines.append('import (')
        lines.append('    "time"')
        lines.append(')')
        lines.append('')

        # 添加注释
        if self.table_comment:
            lines.append(f'// {struct_name} {self.table_comment}')
        else:
            lines.append(f'// {struct_name} {self.table_name}的模型')
        lines.append(f'type {struct_name} struct {{')

        # 添加字段
        for col in self.columns:
            if col.comment:
                lines.append(f'    // {col.comment}')
            lines.append(f'    {col.go_field_name.ljust(20)} {col.go_type.ljust(15)} {col.json_tag.ljust(30)} {col.gorm_tag}')
            lines.append('')

        lines.append('}')
        lines.append('')

        # 使用表名首字母作为接收器
        receiver = struct_name[0].lower()
        if struct_name[0] == 'I' and len(struct_name) > 1 and struct_name[1].isupper():
            # 如果是Id开头，使用k
            receiver = 'k'
        elif len(struct_name) <= 1:
            receiver = 'm'

        lines.append(f'func ({receiver} *{struct_name}) TableName() string {{')
        lines.append(f'    return "{self.table_name}"')
        lines.append('}')

        return '\n'.join(lines)

    def _table_name_to_struct_name(self, table_name: str) -> str:
        """将表名转换为结构体名（大驼峰）"""
        # 移除表前缀（可选）
        # 例如：kk_defense_bandwidth_bill_daily -> DefenseBandwidthBillDaily
        parts = table_name.split('_')

        # 如果第一个部分很短或看起来像前缀，可以跳过
        if len(parts) > 1 and len(parts[0]) <= 3:
            parts = parts[1:]

        # 转换为大驼峰
        return ''.join(x.capitalize() for x in parts)


def parse_sql_string(sql: str, package_name: str = 'models') -> str:
    """解析SQL字符串并返回Go代码"""
    parser = MySQLSQLParser()

    if not parser.parse_sql(sql):
        return ""

    return parser.generate_go_model(package_name)


def parse_multiple_tables(sql: str, package_name: str = 'models') -> Dict[str, str]:
    """解析SQL字符串中的多个表并返回Go代码字典"""
    results = {}

    # 使用正则表达式分割多个CREATE TABLE语句
    # 匹配 CREATE TABLE 开头到结尾的完整语句（包括结尾的分号）
    table_pattern = re.compile(r'CREATE\s+TABLE[^;]*?ENGINE[^;]*?(?:;|$)',
                               re.IGNORECASE | re.DOTALL)

    matches = table_pattern.finditer(sql)

    for match in matches:
        table_sql = match.group(0)
        parser = MySQLSQLParser()

        if parser.parse_sql(table_sql):
            go_code = parser.generate_go_model(package_name)
            if go_code:
                results[parser.table_name] = {
                    'go_code': go_code,
                    'struct_name': parser._table_name_to_struct_name(parser.table_name),
                    'table_comment': parser.table_comment
                }

    return results


def main():
    parser = argparse.ArgumentParser(description='MySQL SQL转Go GORM模型')
    parser.add_argument('sql', nargs='?', help='SQL语句')
    parser.add_argument('--file', help='从文件读取SQL')
    parser.add_argument('--stdin', action='store_true', help='从标准输入读取SQL')
    parser.add_argument('--package', default='models', help='Go包名')
    parser.add_argument('--output', help='输出文件路径（默认输出到标准输出）')
    parser.add_argument('--output-dir', help='输出目录路径（当有多个表时，每个表生成一个文件）')
    parser.add_argument('--multiple', action='store_true', help='强制解析多个表（即使只有一个表）')

    args = parser.parse_args()

    # 获取SQL内容
    sql = ""

    if args.stdin:
        print("请输入SQL语句（按Ctrl+D结束）：")
        sql = sys.stdin.read()
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                sql = f.read()
        except Exception as e:
            print(f"✗ 读取文件失败: {e}")
            return 1
    elif args.sql:
        sql = args.sql
    else:
        print("请提供SQL语句、文件或使用--stdin")
        parser.print_help()
        return 1

    # 判断是否解析多个表
    if args.multiple or args.file or args.stdin:
        # 尝试解析多个表
        results = parse_multiple_tables(sql, args.package)

        if len(results) == 1 and not args.multiple:
            # 只有一个表，直接输出
            table_name = list(results.keys())[0]
            go_code = results[table_name]['go_code']

            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(go_code)
                print(f"✓ Go模型已保存到: {args.output}")
            else:
                print("\n" + "="*80)
                print("Generated Go Model:")
                print("="*80)
                print(go_code)
        elif len(results) > 0:
            # 多个表
            if args.output_dir:
                import os
                os.makedirs(args.output_dir, exist_ok=True)

                for table_name, data in results.items():
                    struct_name = data['struct_name']
                    go_code = data['go_code']
                    filename = f"{struct_name}.go"
                    filepath = os.path.join(args.output_dir, filename)

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(go_code)
                    print(f"✓ {table_name} -> {filepath}")
            else:
                # 输出到标准输出
                for i, (table_name, data) in enumerate(results.items()):
                    if i > 0:
                        print("\n" + "="*80)
                    print(f"\nTable: {table_name}")
                    if data['table_comment']:
                        print(f"Comment: {data['table_comment']}")
                    print("\n" + data['go_code'])

                print(f"\n\n✓ 共转换了 {len(results)} 个表")
        else:
            print("✗ 没有找到有效的表定义")
            return 1
    else:
        # 解析单个表
        go_code = parse_sql_string(sql, args.package)

        if not go_code:
            print("✗ 解析SQL失败")
            return 1

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(go_code)
            print(f"✓ Go模型已保存到: {args.output}")
        else:
            print("\n" + "="*80)
            print("Generated Go Model:")
            print("="*80)
            print(go_code)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())