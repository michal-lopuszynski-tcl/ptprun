MODULE_NAME=ptprun

PY_DIRS=src/ptprun tests setup.py

#PY_MYPY_FLAKE8=src/ptprun tests setup.py

PY_MYPY_FLAKE8=src/ptprun setup.py


FILES_TO_CLEAN=src/ptprun.egg-info dist

include Makefile.inc
