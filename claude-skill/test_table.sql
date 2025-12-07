CREATE TABLE `kk_defense_bandwidth_bill_month` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增ID',
  `bill_total_id` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '所属总账单ID',
  `user_id` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '会员ID',
  `periods` varchar(20) NOT NULL DEFAULT '' COMMENT '账期(格式: YYYY-MM)',
  `product_name` varchar(255) NOT NULL DEFAULT '' COMMENT '产品名称',
  `scheme_type` tinyint(2) unsigned NOT NULL DEFAULT '0' COMMENT '计费项目: 1-带宽 2-IP',
  `bill_days` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '计费周期(天数)',
  `bill_content` text COMMENT '计费详情(JSON格式)',
  `money_content` text COMMENT '报价方案(JSON格式)',
  `money_total` decimal(10,2) unsigned NOT NULL DEFAULT '0.00' COMMENT '费用',
  `create_type` tinyint(2) unsigned NOT NULL DEFAULT '1' COMMENT '生成方式: 1-系统自动 2-手动创建',
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `is_delete` tinyint(2) NOT NULL COMMENT '删除状态 1否 2是',
  PRIMARY KEY (`id`),
  KEY `idx_bill_total_id` (`bill_total_id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_periods` (`periods`),
  KEY `idx_scheme_type` (`scheme_type`),
  KEY `idx_create_time` (`create_time`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COMMENT='防护带宽月度账单明细表';




-- goblog.blog_post definition

CREATE TABLE `blog_post` (
  `pid` int NOT NULL AUTO_INCREMENT,
  `title` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `content` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `markdown` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `category_id` int NOT NULL,
  `user_id` bigint NOT NULL,
  `view_count` int NOT NULL DEFAULT '0',
  `type` int NOT NULL DEFAULT '0',
  `slug` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `create_at` datetime NOT NULL,
  `update_at` datetime NOT NULL,
  PRIMARY KEY (`pid`) USING BTREE,
  UNIQUE KEY `idx_pid` (`pid`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=29 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci ROW_FORMAT=DYNAMIC;