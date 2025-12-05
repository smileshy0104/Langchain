-- ==============================
-- 数据库: blog_db
-- ==============================

-- 用户表
CREATE TABLE `blog_db`.`users` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `username` varchar(50) NOT NULL COMMENT '用户名',
  `email` varchar(100) NOT NULL COMMENT '邮箱',
  `password_hash` varchar(255) NOT NULL COMMENT '密码哈希',
  `avatar_url` varchar(500) DEFAULT NULL COMMENT '头像URL',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否激活',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_username` (`username`),
  UNIQUE KEY `idx_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- ==============================
-- 数据库: shop_db
-- ==============================

-- 商品分类表
CREATE TABLE `shop_db`.`categories` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT COMMENT '分类ID',
  `parent_id` int(11) unsigned DEFAULT NULL COMMENT '父分类ID',
  `name` varchar(100) NOT NULL COMMENT '分类名称',
  `description` text COMMENT '分类描述',
  `sort_order` int(11) NOT NULL DEFAULT '0' COMMENT '排序',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_parent_id` (`parent_id`),
  KEY `idx_sort_order` (`sort_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品分类表';

-- 商品表
CREATE TABLE `shop_db`.`products` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '商品ID',
  `category_id` int(11) unsigned NOT NULL COMMENT '分类ID',
  `sku` varchar(100) NOT NULL COMMENT '商品SKU',
  `name` varchar(200) NOT NULL COMMENT '商品名称',
  `description` longtext COMMENT '商品描述',
  `price` decimal(10,2) unsigned NOT NULL COMMENT '价格',
  `stock` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '库存数量',
  `sales_count` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '销量',
  `image_urls` json DEFAULT NULL COMMENT '商品图片URLs',
  `attributes` json DEFAULT NULL COMMENT '商品属性',
  `status` tinyint(2) NOT NULL DEFAULT '1' COMMENT '状态：0-下架，1-上架',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `idx_sku` (`sku`),
  KEY `idx_category_id` (`category_id`),
  KEY `idx_status` (`status`),
  KEY `idx_price` (`price`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品表';

-- ==============================
-- 数据库: forum_db
-- ==============================

-- 版块表
CREATE TABLE `forum_db`.`forums` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT COMMENT '版块ID',
  `name` varchar(100) NOT NULL COMMENT '版块名称',
  `description` varchar(500) DEFAULT NULL COMMENT '版块描述',
  `parent_id` int(11) unsigned DEFAULT '0' COMMENT '父版块ID',
  `topic_count` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '主题数量',
  `post_count` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '帖子数量',
  `last_post_id` bigint(20) unsigned DEFAULT NULL COMMENT '最后帖子ID',
  `last_post_time` datetime DEFAULT NULL COMMENT '最后发帖时间',
  `sort_order` int(11) NOT NULL DEFAULT '0' COMMENT '排序',
  `is_active` tinyint(1) NOT NULL DEFAULT '1' COMMENT '是否启用',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_parent_id` (`parent_id`),
  KEY `idx_sort_order` (`sort_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='版块表';

-- 主题表
CREATE TABLE `forum_db`.`topics` (
  `id` bigint(20) unsigned NOT NULL AUTO_INCREMENT COMMENT '主题ID',
  `forum_id` int(11) unsigned NOT NULL COMMENT '版块ID',
  `user_id` int(11) unsigned NOT NULL COMMENT '用户ID',
  `title` varchar(200) NOT NULL COMMENT '主题标题',
  `content` longtext NOT NULL COMMENT '主题内容',
  `view_count` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '浏览次数',
  `reply_count` int(11) unsigned NOT NULL DEFAULT '0' COMMENT '回复数量',
  `is_top` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否置顶',
  `is_locked` tinyint(1) NOT NULL DEFAULT '0' COMMENT '是否锁定',
  `last_reply_id` bigint(20) unsigned DEFAULT NULL COMMENT '最后回复ID',
  `last_reply_time` datetime DEFAULT NULL COMMENT '最后回复时间',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_forum_id` (`forum_id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_is_top` (`is_top`),
  KEY `idx_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='主题表';