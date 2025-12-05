package models

import (
    "time"
)

// DefenseBandwidthBillMonth 防护带宽月度账单明细表
type DefenseBandwidthBillMonth struct {
    // 自增ID
    Id                   uint32          `json:"id"`                    `gorm:"column:id;AUTO_INCREMENT;NOT NULL;comment:'自增ID'"`

    // 所属总账单ID
    billTotalId          uint32          `json:"bill_total_id"`         `gorm:"column:bill_total_id;NOT NULL;default:0;comment:'所属总账单ID'"`

    // 会员ID
    userId               uint32          `json:"user_id"`               `gorm:"column:user_id;NOT NULL;default:0;comment:'会员ID'"`

    // 账期(格式: YYYY-MM)
    periods              string          `json:"periods"`               `gorm:"column:periods;NOT NULL;default:;comment:'账期(格式: YYYY-MM)'"`

    // 产品名称
    productName          string          `json:"product_name"`          `gorm:"column:product_name;NOT NULL;default:;comment:'产品名称'"`

    // 计费项目: 1-带宽 2-IP
    schemeType           uint8           `json:"scheme_type"`           `gorm:"column:scheme_type;NOT NULL;default:0;comment:'计费项目: 1-带宽 2-IP'"`

    // 计费周期(天数)
    billDays             uint32          `json:"bill_days"`             `gorm:"column:bill_days;NOT NULL;default:0;comment:'计费周期(天数)'"`

    // 计费详情(JSON格式)
    billContent          string          `json:"bill_content"`          `gorm:"column:bill_content;comment:'计费详情(JSON格式)'"`

    // 报价方案(JSON格式)
    moneyContent         string          `json:"money_content"`         `gorm:"column:money_content;comment:'报价方案(JSON格式)'"`

    // 费用
    moneyTotal           float64         `json:"money_total"`           `gorm:"column:money_total;NOT NULL;default:0.00;type:decimal(10,2) unsigned;comment:'费用'"`

    // 生成方式: 1-系统自动 2-手动创建
    createType           uint8           `json:"create_type"`           `gorm:"column:create_type;NOT NULL;default:1;comment:'生成方式: 1-系统自动 2-手动创建'"`

    // 创建时间
    createTime           time.Time       `json:"create_time"`           `gorm:"column:create_time;NOT NULL;default:CURRENT_TIMESTAMP;type:timestamp;comment:'创建时间'"`

    // 更新时间
    updateTime           time.Time       `json:"update_time"`           `gorm:"column:update_time;NOT NULL;default:CURRENT_TIMESTAMP;type:timestamp;comment:'更新时间'"`

    // 删除状态 1否 2是
    isDelete             int8            `json:"is_delete"`             `gorm:"column:is_delete;NOT NULL;comment:'删除状态 1否 2是'"`

}

func (d *DefenseBandwidthBillMonth) TableName() string {
    return "kk_defense_bandwidth_bill_month"
}