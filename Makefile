.PHONY: install test smoke data ablation benchmark report explain clean

install:
	pip install -e ".[tune,dev]"

test:
	pytest -q

smoke:            ## end-to-end check on synthetic data (CPU, ~1 min)
	python scripts/run_suite.py --suite configs/suites/smoke.yaml
	python scripts/report.py --suite configs/suites/smoke.yaml --n-boot 100

data:
	python scripts/prepare_data.py --sizes 28 224 --download

ablation:         ## Phase 2: tune the from-scratch CNN
	python scripts/run_ablation.py --plan configs/ablation/cnn_ablation.yaml

benchmark:        ## Phase 4: E1 + E3
	python scripts/run_suite.py --suite configs/suites/e1_e3_benchmark.yaml

report:
	python scripts/report.py --suite configs/suites/e1_e3_benchmark.yaml

clean:
	rm -rf runs/smoke reports/smoke .pytest_cache
