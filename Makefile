all: install

.PHONY: clean
clean:
	python setup.py clean --all

.PHONY: install
install:
	pip install -e .

.PHONY: kill
kill:
	kill $(ps u | grep '[p]ython' | awk '{print $2}')
