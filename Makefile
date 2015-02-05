all:
	python src/setup.py build_ext --inplace
debug:
	python src/setup.py build_ext --inplace
install:
	python src/setup.py install
clean:
	rm -f *.so
	rm -rf build
