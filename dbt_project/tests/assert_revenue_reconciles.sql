-- The mart must reconcile to completed orders independently of the customer
-- dimension. A one-to-many customer join would fail this assertion.
with expected as (
    select
        order_date,
        count(*) as expected_completed_order_rows,
        sum(amount_usd) as expected_daily_revenue
    from {{ ref('stg_orders') }}
    where status = 'completed'
    group by order_date
),
actual as (
    select order_date, completed_order_rows, daily_revenue
    from {{ ref('fct_daily_revenue') }}
)
select
    coalesce(e.order_date, a.order_date) as order_date,
    e.expected_completed_order_rows,
    a.completed_order_rows,
    e.expected_daily_revenue,
    a.daily_revenue
from expected e
full outer join actual a using (order_date)
where e.order_date is null
   or a.order_date is null
   or e.expected_completed_order_rows <> a.completed_order_rows
   or abs(e.expected_daily_revenue - a.daily_revenue) > 0.0001
