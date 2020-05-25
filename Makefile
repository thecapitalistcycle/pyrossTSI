PYTHON=python
#PYTHON=path to your python installation
path=examples
recursive=True

make:
	@echo Installing pyrosstsi...
	${PYTHON} setup.py install

clean-local:
	@echo removing local compiled files
	rm pyrosstsi/*.c pyrosstsi/*.html

clean:
	@echo removing all compiled files
	${PYTHON} setup.py clean
	rm pyrosstsi/*.c pyrosstsi/*.html
	
env:
	@echo creating conda environment...
	conda env create --file environment.yml
	# conda activate pyrosstsi
	@echo use make to install pyrosstsi

test:
	@echo testing pyrosstsi...
	cd tests && python quick_test.py

nbtest:
	@echo testing example notebooks...
	@echo test $(path)
	cd tests && python notebook_test.py --path $(path) --recursive $(recursive)
