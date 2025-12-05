#!/usr/bin/env python3
"""Initialization script for sql-to-gorm skill setup"""

import os
import sys

def init_skill():
    """Initialize the skill directory structure"""

    # Create necessary directories
    os.makedirs("scripts", exist_ok=True)
    os.makedirs("references", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    # Create example reference file
    with open("references/type_mapping.md", "w") as f:
        f.write("""# MySQL to Go Type Mapping

## Basic Types

### Integers
- `tinyint` → `int8`
- `smallint` → `int16`
- `mediumint` → `int32`
- `int` → `int32`
- `bigint` → `int64`

### Unsigned Integers
- `tinyint unsigned` → `uint8`
- `smallint unsigned` → `uint16`
- `int unsigned` → `uint32`
- `bigint unsigned` → `uint64`

### Floating Point
- `float` → `float32`
- `double` → `float64`
- `decimal(p,s)` → `float64`

### Strings
- `char(n)` → `string`
- `varchar(n)` → `string`
- `text` → `string`
- `longtext` → `string`

### Date/Time
- `date` → `time.Time`
- `datetime` → `time.Time`
- `timestamp` → `time.Time`

### Special
- `tinyint(1)` → `bool`
- `json` → `string` (or map[string]interface{})
- `blob` → `[]byte`
""")

    # Create example asset template
    with open("assets/template.go", "w") as f:
        f.write("""package models

import (
    "time"
)

// {{.StructName}} {{.TableComment}}
type {{.StructName}} struct {
{{range .Fields}}
    // {{.Comment}}
    {{.FieldName}} {{.Type}} `json:"{{.JsonName}}"` gorm:"{{.GormTag}}"
{{end}}
}

func ({{.Receiver}} *{{.StructName}}) TableName() string {
    return "{{.TableName}}"
}
""")

    print("✓ Skill initialization complete!")
    print("  - Created required directories")
    print("  - Added reference documentation")
    print("  - Added Go template asset")

if __name__ == "__main__":
    init_skill()