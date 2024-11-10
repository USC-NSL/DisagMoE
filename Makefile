all: pip

.PHONY: clean
clean:
	- python setup.py clean --all
	- rm *.so
	- rm -rf csrc/build

.PHONY: pip
pip:
	pip install -e .

.PHONY: cmake
cmake:
	mkdir -p csrc/build
	cd csrc/build && cmake -E .. && make -j && make install

.PHONY: kill
kill:
	kill $(ps u | grep '[p]ython' | awk '{print $2}')
