# Minimal makefile for Sphinx documentation
#

export LC_ALL ?= C.UTF-8

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
# Fallback to local virtual environment binary if sphinx-build is not in system PATH
SPHINXBUILD   ?= $(shell which sphinx-build 2>/dev/null || (test -f ../venv/bin/sphinx-build && echo ../venv/bin/sphinx-build) || echo sphinx-build)
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
