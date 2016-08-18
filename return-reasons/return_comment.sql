--
-- @Author: shubham.chandel
-- @Date:   2016-07-05 14:48:32
-- @Last Modified by:   shubham.chandel
-- @Last Modified time: 2016-07-05 14:52:07
--

SELECT	product_title,
		return_product_category_name,
		final_adjusted_amount,
		return_comments,
		return_reason,
		reason_key,
		oms_qc_reason,
		refund_reason,
		return_sub_reason,
		return_from_address_city,
		courier_name,
		order_item_status
FROM 	bigfoot_external_neo.scp_rrr__return_l2_id_level_hive_fact
WHERE	return_comments IS NOT NULL
LIMIT	1000000;

-- 6092241/10674974 Rows

