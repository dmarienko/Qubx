set windows-shell := ["C:\\Program Files\\Git\\bin\\sh.exe", "-c"]

# publish *FLAGS:
# git push origin master --follow-tags

help:
	@just --list --unsorted


test:
	poetry run pytest


build:
	rm -rf build
	find src -type f -name *.pyd -exec  rm {} \;
	poetry build


dev-install:
	# - install in dev environment
	pip install . --upgrade
	

publish: build test
	@if [ "$(git symbolic-ref --short -q HEAD)" = "master" ]; then rm -rf dist && rm -rf build && poetry build && twine upload dist/*; else echo ">>> Not in master branch !"; fi