APP=libdeep-python
PREFIX?=/usr/local

all:
	python src/setup.py build_ext --inplace
debug:
	python src/setup.py build_ext --inplace
install:
	python src/setup.py install
	mkdir -m 755 -p ${DESTDIR}${PREFIX}/share/man/man1
	install -m 644 man/${APP}.1.gz ${DESTDIR}${PREFIX}/share/man/man1
uninstall:
	rm -f ${PREFIX}/share/man/man1/${APP}.1.gz
clean:
	rm -f *.so
	rm -rf build
