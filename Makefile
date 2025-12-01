PYTHON ?= python3
VENV ?= venv
ACTIVATE = source $(VENV)/bin/activate

.PHONY: install prepare train-svm train-cnn train-all predict test reproduce

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) -m genre_classifier prepare

train-svm:
	$(PYTHON) -m genre_classifier train --target svm --prepare

train-cnn:
	$(PYTHON) -m genre_classifier train --target cnn --prepare

train-all:
	$(PYTHON) -m genre_classifier train --target both --prepare

predict:
	$(PYTHON) -m genre_classifier predict $(AUDIO)

test:
	$(PYTHON) -m pytest -q

reproduce:
	./scripts/reproduce.sh
