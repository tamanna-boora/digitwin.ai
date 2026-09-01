PYTHON := ./.venv/Scripts/python.exe

.PHONY: demo generate train backtest app

demo: generate train backtest

generate:
	$(PYTHON) scripts/generate_data.py

train:
	$(PYTHON) scripts/train_models.py

backtest:
	$(PYTHON) scripts/run_backtest.py

app:
	$(PYTHON) -m streamlit run app/Home.py
