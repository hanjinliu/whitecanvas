set shell := ["powershell.exe", "-c"]

doc:
	sphinx-apidoc -f -o ./rst/apidoc ./whitecanvas
	sphinx-build -b html ./rst ./docs

watch-rst:
	watchfiles "sphinx-build -b html ./rst ./_docs_temp" rst

remove-cache:
	#!python
	from pathlib import Path

	for path in Path(".").glob("**/__pycache__/*"):
		path.unlink(missing_ok=False)
