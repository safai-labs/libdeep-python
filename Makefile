APP=libdeep-python
PREFIX?=/usr/local

all:
	python3 src/setup.py build_ext --inplace
debug:
	python3 src/setup.py build_ext --inplace
install:
	python3 src/setup.py install
	mkdir -m 755 -p ${DESTDIR}${PREFIX}/share/man/man1
	install -m 644 man/${APP}.1.gz ${DESTDIR}${PREFIX}/share/man/man1
uninstall:
	rm -f ${PREFIX}/share/man/man1/${APP}.1.gz
clean:
	rm -f *.so
	rm -rf build
