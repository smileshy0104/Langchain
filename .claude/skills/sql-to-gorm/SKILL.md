---
name: sql-to-gorm
description: This skill should be used when users need to convert MySQL SQL CREATE TABLE statements to Go GORM models. It supports natural language prompts to identify SQL files, extract SQL content, and generate Go struct definitions with proper GORM tags.
---

# SQL to Go GORM Model Converter

This skill converts MySQL CREATE TABLE statements into Go language struct definitions with GORM tags. It supports multiple input methods including direct SQL text, SQL files, and natural language descriptions.

## When to Use This Skill

Use this skill when users:
- Need to generate Go GORM models from MySQL table definitions
- Want to convert SQL DDL to Go structs with proper type mapping
- Have multiple SQL tables to convert efficiently
- Need to handle complex SQL features like unsigned types, constraints, and comments

## Core Workflow

### 1. Parse User Intent

Analyze the user's natural language prompt to identify:
- SQL source (direct text or file path)
- Output preferences (display or save to files)
- Package name and output directory
- Single table vs multiple tables

### 2. Extract SQL Content

Retrieve SQL using the appropriate method:
- Extract CREATE TABLE statements from conversation
- Load SQL from specified file paths
- Handle database prefixes and multiple schemas
- Clean and normalize SQL syntax

### 3. Generate Go Models

Execute `sql_to_gorm_skill.py` with parsed intent:
- Convert MySQL types to appropriate Go types
- Handle unsigned integers and special types
- Generate complete GORM tags with constraints
- Preserve table and column comments

### 4. Present Results

Format output based on user preferences:
- Display generated code in conversation
- Save to specified directory with proper naming
- Create summary of generated files
- Report any warnings or errors

## Key Features

### Type Mapping Support
- Unsigned integers (uint8, uint16, uint32, uint64)
- Standard MySQL types (INT, VARCHAR, TEXT, etc.)
- Special types (JSON, TIMESTAMP, DECIMAL)
- NULL handling with pointer types

### Advanced SQL Features
- Database prefixes (db_name.table_name)
- Multiple CREATE TABLE statements
- Table and column comments
- Constraints and indexes
- Default values and AUTO_INCREMENT

### Output Options
- Single file with all models
- Individual files per table
- Custom package names
- Custom output directories

## Using the Skill

The skill processes natural language prompts to convert SQL to Go models:

### Direct SQL Conversion

```python
# Use the skill wrapper for interactive processing
from scripts.sql_to_gorm_skill import SQLToGORMSkill

skill = SQLToGORMSkill()
result = skill.process("Convert this SQL to Go: CREATE TABLE users (id int primary key)")
```

### Natural Language Examples

- "Convert my schema.sql file to Go models in ./models directory"
- "I have SQL for a blog system, please generate GORM models"
- "Convert these tables to Go with package name 'entities'"
- "Parse the database export and create separate files for each table"

### Intent Recognition

The skill automatically detects:
- SQL statements in the conversation
- File paths and output directories
- Package names and multiple table preferences
- Database prefixes and complex schemas

### File-based Processing

```bash
# Interactive processing with natural language
python scripts/sql_to_gorm_skill.py "Convert schema.sql to models in ./internal"

# Direct command line usage
python scripts/sql_to_gorm.py --file schema.sql --output-dir ./models
```

## Error Handling

- Invalid SQL syntax detection
- Missing file handling
- Permission errors
- Type conversion warnings
- File creation errors

## Best Practices

1. Always validate SQL syntax before conversion
2. Check that output directories exist or can be created
3. Review generated code for custom type adjustments
4. Test generated models with GORM operations
5. Keep original SQL as documentation

## Integration Examples

### In Go Project

```go
// Example using generated models
db.AutoMigrate(&User{}, &Product{}, &Order{})

var users []User
db.Find(&users)
```

### With Build Systems

```bash
# Generate models as part of build
generate:
	python scripts/sql_to_gorm.py --file schema.sql --output-dir ./models
	go build ./...
```