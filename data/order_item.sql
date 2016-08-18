--
-- @Author: shubham.chandel
-- @Date:   2016-07-05 14:48:32
-- @Last Modified by:   shubham.chandel
-- @Last Modified time: 2016-07-05 14:52:07
--

SELECT	order_item_title,
		order_date_time
FROM 	bigfoot_external_neo.scp_oms__order_item_unit_s1_365_final_fact
WHERE	order_item_category_id == 20144;

-- 6092241/10674974 Rows

