PYTHON ?= python

.PHONY: reset baseline tests adversarial tests-all gx dbt dashboard generate verify

reset:
	$(PYTHON) scripts/reset_lab.py

baseline:
	$(PYTHON) scripts/run_baseline.py

tests:
	$(PYTHON) -m pytest tests_public -q

adversarial:
	$(PYTHON) -m pytest tests_adversarial -q

tests-all: tests adversarial

gx:
	$(PYTHON) gx/validate_orders.py

dbt:
	$(PYTHON) scripts/sync_dbt_seeds.py
	dbt build --project-dir dbt_project --profiles-dir dbt_project

dashboard:
	streamlit run dashboard/app.py

generate:
	$(PYTHON) scripts/generate_data.py --rows 600 --days 42 --seed 27

verify: reset baseline tests-all gx dbt
