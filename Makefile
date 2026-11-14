.PHONY: setup check explore test train-quick train evaluate rollouts videos plots report viewer first-run all

setup:
	pip install -r requirements.txt
	pip install -e .

check:
	python scripts/00_check_install.py

explore:
	python scripts/01_random_agent.py
	python scripts/02_explain_reward.py

test:
	python -m pytest tests/ -v

train-quick:
	python -m ll.train --config baseline --seed 0 --quick

# Seeds of one config run concurrently (3 jobs); configs run one after
# another so at most 3 trainings share the CPU at a time. Each seed writes
# only its own runs/<config>/seed<seed>/ directory, so there is no clash.
# On a low-core machine, drop the "&" and the "wait" to go back to sequential.
train:
	for c in baseline time_penalty hover_hack fixed; do \
		for s in 0 1 2; do \
			python -m ll.train --config $$c --seed $$s & \
		done; \
		wait; \
	done

evaluate:
	python -m ll.evaluate

rollouts:
	python -m ll.rollouts

videos:
	python -m ll.video

plots:
	python -m ll.plots

report:
	python -m ll.report

viewer:
	python -m ll.live_viewer --config baseline --seed 0

first-run: check explore test train-quick evaluate plots report
	@echo ">>> First run complete!"

all: check explore test train evaluate rollouts plots report

