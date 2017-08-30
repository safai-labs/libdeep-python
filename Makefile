APP=libdeep-python
VERSION=1.00
PREFIX?=/usr/local
CURR_DIR=$(shell pwd)
SELF_DIR=$(shell basename $(CURR_DIR))

DATE_FMT = %Y-%m-%d
ifdef SOURCE_DATE_EPOCH
	BUILD_DATE ?= $(shell date -u -d "@$(SOURCE_DATE_EPOCH)" "+$(DATE_FMT)"  2>/dev/null || date -u -r "$(SOURCE_DATE_EPOCH)" "+$(DATE_FMT)" 2>/dev/null || date -u "+$(DATE_FMT)")
else
	BUILD_DATE ?= $(shell date "+$(DATE_FMT)")
endif

all:
	python3 src/setup.py build_ext --inplace
debug:
	python3 src/setup.py build_ext --inplace
source:
	tar -cvf ../${APP}_${VERSION}.orig.tar --exclude-vcs ../$(SELF_DIR)
	gzip -f9n ../${APP}_${VERSION}.orig.tar
install:
	python3 src/setup.py install
	mkdir -m 755 -p ${DESTDIR}${PREFIX}/share/man/man1
	install -m 644 man/${APP}.1.gz ${DESTDIR}${PREFIX}/share/man/man1
uninstall:
	rm -f ${PREFIX}/share/man/man1/${APP}.1.gz
clean:
	rm -rf build
	rm -f *.so
