-- Keep exactly one active dimension version per customer before joining. A
-- duplicate active row must not multiply a completed order or its revenue.

with completed_orders as (
    select *
    from {{ ref('stg_orders') }}
    where status = 'completed'
),
ranked_active_customers as (
    select
        customer_id,
        row_number() over (
            partition by customer_id
            order by valid_from desc nulls last
        ) as active_version_rank
    from {{ ref('stg_customers') }}
    where is_active = true
),
active_customers as (
    select customer_id
    from ranked_active_customers
    where active_version_rank = 1
)
select
    o.order_date,
    count(*) as completed_order_rows,
    sum(o.amount_usd) as daily_revenue
from completed_orders o
left join active_customers c
    on o.customer_id = c.customer_id
group by 1
order by 1
