devlop:
	pip install -ve .

install:
	pip install .

requirements:
	pip install -r requirements.txt

clean:
	rm -rf build dist *.egg-info
	pip uninstall visloc_localization -y