CC=g++
CFLAGS=-Wall -std=c++17 -O3 -funroll-loops -DASIO_STANDALONE

INCLUDES=-Iinclude #-I/opt/homebrew/opt/openssl@1.1/include/


# Uncomment If throws an error, and not using python3
PYTHON_INCLUDE_PATH=$(shell python -c "import sysconfig; print(sysconfig.get_path('include'))")
LIBRARYPATH=$(shell python -c "import sysconfig; import os;print(os.path.split(os.path.split(sysconfig.get_path('platlib'))[0])[0])")
PYTHON_VERSION=$(shell python -c "import sys; print('python' + '.'.join(sys.version.split('|')[0].strip().split('.')[:2]))")
# export LIBRARY_PATH=$(LIBRARYPATH):$(LD_LIBRARY_PATH)


rollerball:
	mkdir -p bin
	$(CC) $(CFLAGS) $(INCLUDES) src/server.cpp src/board.cpp src/engine.cpp src/rollerball.cpp src/uciws.cpp -lpthread -o bin/rollerball

rollerball_py:
	mkdir -p bin
	pip install -e .
	LIBRARY_PATH=$(LIBRARYPATH) $(CC) $(CFLAGS) $(INCLUDES) -Wl,-rpath,$(LIBRARYPATH) `python3 -m pybind11 --includes` src/server.cpp src/board.cpp src/engine_py.cpp src/rollerball.cpp src/uciws.cpp -o bin/rollerball_py -I$(PYTHON_INCLUDE_PATH) -lpthread -l$(PYTHON_VERSION) -fPIC

package:
	mkdir -p build
	rm -rf build/*
	mkdir build/rollerball build/rollerball/src
	cp -r include build/rollerball/include
	cp src/*.hpp build/rollerball/src/
	cp src/board.cpp src/bindings.cpp src/engine.cpp src/engine_py.cpp src/rollerball.cpp src/server.cpp src/uciws.cpp build/rollerball/src/
	cp -r scripts build/rollerball/scripts
	cp engine.py setup.py build/rollerball/
	cp Makefile build/rollerball/
	cd web && npm run build
	cp -r web/dist build/rollerball/web
	cd build && zip -r rollerball.zip rollerball

dbg_frontend: src/debug_frontend.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) src/server.cpp src/debug_frontend.cpp -o bin/debug_frontend

dbg_uciws: src/debug_uciws.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) src/server.cpp src/uciws.cpp src/debug_uciws.cpp -o bin/debug_uciws

dbg_board: src/debug_board.cpp 
	$(CC) $(CFLAGS) $(INCLUDES) src/board.cpp src/debug_board.cpp -o bin/debug_board

clean:
	rm bin/*
