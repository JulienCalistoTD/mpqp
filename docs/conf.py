from __future__ import annotations
from typing import Literal
import dotenv

# Configuration file for the Sphinx documentation builder.
from sphinx.application import Sphinx
from sphinx.highlighting import PygmentsBridge
from pygments.formatters.latex import LatexFormatter

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MPQP"
copyright = "2024, ColibrITD"
package_name = ""
author = "ColibrITD"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx_rtd_dark_mode",
    "sphinx_copybutton",
]
default_dark_mode = True
autodoc_typehints = "description"
autodoc_type_aliases = {"Matrix": "Matrix", "AvailableDevice": "AvailableDevice"}
simplify_optional_unions = True
typehints_defaults = "comma"
dotenv.load_dotenv()
sphinx_github_changelog_token = os.getenv("SPHINX_GITHUB_CHANGELOG_TOKEN")
if sphinx_github_changelog_token is not None:
    extensions.append("sphinx_github_changelog")

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "math"

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

pythonversion = sys.version.split(" ")[0]
# Python and Sage trac ticket shortcuts. For example, :trac:`7549` .
extlinks = {
    "python": ("https://docs.python.org/release/" + pythonversion + "/%s", ""),
    "trac": ("http://trac.sagemath.org/%s", "trac ticket #"),
    "wikipedia": ("https://en.wikipedia.org/wiki/%s", "Wikipedia article "),
    "arxiv": ("http://arxiv.org/abs/%s", "Arxiv "),
    "oeis": ("https://oeis.org/%s", "OEIS sequence "),
    "doi": ("https://dx.doi.org/%s", "doi:"),
    "mathscinet": ("http://www.ams.org/mathscinet-getitem?mr=%s", "MathSciNet "),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = "alabaster"
# html_theme = "furo"  # pip install furo
# html_theme = "sphinx_rtd_theme"  # pip install sphinx-rtd-theme
html_context = {
    "display_github": True,
    "github_repo": "ColibrITD-SAS/mpqp",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ["custom.css"]
# html_css_files = []

html_js_files = ["custom.js"]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "MPQP"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "resources/mpqp-logo-dark-theme.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "resources/favicon.ico"

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = package_name + "doc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    "preamble": "",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        "index",
        package_name + ".tex",
        "Documentation of " + package_name,
        author,
        "manual",
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", package_name, package_name + " documentation", [author], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        package_name,
        package_name + " documentation",
        author,
        package_name,
        project,
        "Miscellaneous",
    ),
]


# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False


# This is to make the verbatim font smaller;
# Verbatim environment is not breaking long lines


class CustomLatexFormatter(LatexFormatter):  # type: ignore
    """Adds custom options to LaTeX outputs"""

    def __init__(self, **options):  # type: ignore
        super(CustomLatexFormatter, self).__init__(**options)
        self.verboptions = r"formatcom=\footnotesize"


PygmentsBridge.latex_formatter = CustomLatexFormatter

latex_elements[
    "preamble"
] += r"""
% One-column index
\makeatletter
\renewenvironment{theindex}{
  \chapter*{\indexname}
  \markboth{\MakeUppercase\indexname}{\MakeUppercase\indexname}
  \setlength{\parskip}{0.1em}
  \relax
  \let\item\@idxitem
}{}
\makeatother
\renewcommand{\ttdefault}{txtt}
"""

autodoc_default_options = {
    "members": None,
    "undoc-members": None,
    "show-inheritance": None,
    # 'private-members': None,
    # 'special-members': '__init__',
    # 'member-order': 'bysource',
    # 'exclude-members': '__weakref__'
}

autodoc_member_order = "groupwise"

WhatType = Literal["module", "class", "exception", "function", "method", "attribute"]
Options = Literal["inherited_members", "undoc_members", "show_inheritance", "no-index"]
OptionsType = dict[Options, str]


def maybe_skip_member(
    app: Sphinx,
    what: WhatType,
    name: str,
    obj: object,
    skip: bool,
    options: OptionsType,
):
    return skip or "M-TODO" in str(obj.__doc__)


def setup(app: Sphinx):
    app.connect("autodoc-skip-member", maybe_skip_member)
