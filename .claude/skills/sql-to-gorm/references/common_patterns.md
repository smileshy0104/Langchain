# Common SQL Patterns and Conversions

## Primary Keys
```sql
id int PRIMARY KEY AUTO_INCREMENT
```
→
```go
Id uint32 `json:"id" gorm:"column:id;primary_key;AUTO_INCREMENT;NOT NULL"`
```

## Foreign Keys
```sql
user_id int(11) unsigned NOT NULL
```
→
```go
UserId uint32 `json:"user_id" gorm:"column:user_id;NOT NULL"`
```

## Timestamps
```sql
created_at timestamp DEFAULT CURRENT_TIMESTAMP
updated_at timestamp DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
```
→
```go
CreatedAt time.Time `json:"created_at" gorm:"column:created_at;default:CURRENT_TIMESTAMP"`
UpdatedAt time.Time `json:"updated_at" gorm:"column:updated_at;default:CURRENT_TIMESTAMP"`
```

## JSON Fields
```sql
metadata json COMMENT 'Additional metadata'
```
→
```go
Metadata string `json:"metadata" gorm:"column:metadata;comment:'Additional metadata'"`
```

## Enum-like Fields
```sql
status tinyint(2) NOT NULL DEFAULT '0' COMMENT '0-inactive,1-active,2-suspended'
```
→
```go
Status int8 `json:"status" gorm:"column:status;NOT NULL;default:0;comment:'0-inactive,1-active,2-suspended'"`
```

## Soft Deletes
```sql
deleted_at timestamp NULL
```
→
```go
DeletedAt *time.Time `json:"deleted_at" gorm:"column:deleted_at"`
```

## Money Fields
```sql
price decimal(10,2) unsigned NOT NULL DEFAULT '0.00'
```
→
```go
Price float64 `json:"price" gorm:"column:price;NOT NULL;default:0.00;type:decimal(10,2) unsigned"`
```

## Boolean Fields
```sql
is_active tinyint(1) NOT NULL DEFAULT '1'
```
→
```go
IsActive bool `json:"is_active" gorm:"column:is_active;NOT NULL;default:1"`
```