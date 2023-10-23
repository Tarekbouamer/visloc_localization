requirements:
	pip install -r requirements.txt

full_requirements:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
	pip install -r requirements.txt

install:
	pip install .

develop:
	pip install -ve .

clean:
	rm -rf build dist *.egg-info
	pip uninstall -y loc

lint:
	ruff check ./loc/ --ignore E501 --quiet 
