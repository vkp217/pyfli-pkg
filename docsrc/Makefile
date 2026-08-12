# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment, for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

clean:
	rm -rf "$(BUILDDIR)" api/generated _build_polyversion

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


.PHONY: local-docs production-docs production-docs-sequential view-docs clean-all

local-docs:
	@sphinx-polyversion --local poly.py
	@echo 'Copying docs to ../docs'
	@rm -rf ../docs/*
	@cp -a _build_polyversion/. ../docs
	@rm -rf _build_polyversion

production-docs:
	@sphinx-polyversion poly.py
	@echo 'Copying docs to ../docs'
	@rm -rf ../docs/*
	@cp -a _build_polyversion/. ../docs
	@rm -rf _build_polyversion

production-docs-sequential:
	@sphinx-polyversion --sequential poly.py
	@echo 'Copying docs to ../docs'
	@rm -rf ../docs/*
	@cp -a _build_polyversion/. ../docs
	@rm -rf _build_polyversion

view-docs:
	@echo "Serving ../docs at http://127.0.0.1:8090 ... Ctrl+C to stop."
	@python -m http.server -d ../docs -b 127.0.0.1 8090

clean-all: clean
	@rm -rf _build_polyversion .docs_venvs
	@rm -rf api/generated
