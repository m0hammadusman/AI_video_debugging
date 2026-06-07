.PHONY: install run models test clean

install:
	python -m pip install -r requirements.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

models:
	python -m scripts.download_models

test:
	pytest -q

clean:
	python -m scripts.cleanup_old_jobs
