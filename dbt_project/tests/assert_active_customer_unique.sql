-- An active SCD dimension must contain at most one row per customer. This is
-- intentionally a singular data test rather than a generic unique test,
-- because historical inactive versions are valid.
select
    customer_id,
    count(*) as active_row_count
from {{ ref('stg_customers') }}
where is_active = true
group by customer_id
having count(*) > 1
