package models

import (
    "time"
)

// BlogPost blog_post的模型
type BlogPost struct {
    pid                  int32           `json:"pid"`                   `gorm:"column:pid;AUTO_INCREMENT;NOT NULL;comment:''"`

    title                string          `json:"title"`                 `gorm:"column:title;NOT NULL;comment:''"`

    content              string          `json:"content"`               `gorm:"column:content;NOT NULL;comment:''"`

    markdown             string          `json:"markdown"`              `gorm:"column:markdown;NOT NULL;comment:''"`

    categoryId           int32           `json:"category_id"`           `gorm:"column:category_id;NOT NULL;comment:''"`

    userId               int64           `json:"user_id"`               `gorm:"column:user_id;NOT NULL;comment:''"`

    viewCount            int32           `json:"view_count"`            `gorm:"column:view_count;NOT NULL;default:0;comment:''"`

    type                 int32           `json:"type"`                  `gorm:"column:type;NOT NULL;default:0;comment:''"`

    slug                 string          `json:"slug"`                  `gorm:"column:slug;default:NULL;comment:''"`

    createAt             time.Time       `json:"create_at"`             `gorm:"column:create_at;NOT NULL;type:timestamp;comment:''"`

    updateAt             time.Time       `json:"update_at"`             `gorm:"column:update_at;NOT NULL;type:timestamp;comment:''"`

}

func (b *BlogPost) TableName() string {
    return "blog_post"
}