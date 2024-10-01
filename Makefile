clean:
	python setup.py clean --all

install:
	pip install .

kill:
	kill $(ps u | grep '[p]ython' | awk '{print $2}')

all: clean install kill
	