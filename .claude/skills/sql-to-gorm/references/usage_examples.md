# SQL to GORM Usage Examples

## Example 1: Single Table Conversion

**User Prompt:**
"Convert this table to GORM: CREATE TABLE users (id int primary key, name varchar(100))"

**Expected Action:**
1. Extract the SQL statement
2. Parse and convert to Go struct
3. Display result

## Example 2: File-based Conversion

**User Prompt:**
"I have my database schema in /path/to/schema.sql, can you convert all tables to Go models in ./models directory?"

**Expected Action:**
1. Read SQL file from specified path
2. Parse all CREATE TABLE statements
3. Generate Go files in output directory
4. Report completion

## Example 3: Multiple Databases

**User Prompt:**
"I have exported my database including multiple schemas. The file is at ~/db_export.sql. Please create separate models for each table."

**Expected Action:**
1. Handle database prefixes (db_name.table_name)
2. Convert all tables to Go structs
3. Keep database context in comments

## Example 4: Complex Schema

**User Prompt:**
"Convert this ecommerce schema with products, orders, and users tables. Pay attention to the unsigned integers and decimal fields."

**Expected Action:**
1. Parse complex schema with relationships
2. Correctly map all MySQL types
3. Preserve all constraints and comments

## Example 5: Interactive Conversion

**User Prompt:**
"I'll paste the SQL for a blog system. Can you generate the models and save them to ./internal/models with package name 'models'?"

**Expected Action:**
1. Accept SQL from user
2. Generate models with custom package name
3. Save to specified directory
4. Provide summary