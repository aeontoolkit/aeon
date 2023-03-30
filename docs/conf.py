#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuration file for the Sphinx documentation builder."""

import os
import sys
from importlib import import_module

import aeon

# -- Project information -----------------------------------------------------

project = "aeon"
copyright = "The aeon developers (BSD-3 License)"
author = "aeon developers"

version = aeon.__version__
release = aeon.__version__

github_tag = f"v{version}"

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

ON_READTHEDOCS = os.environ.get("READTHEDOCS") == "True"
if not ON_READTHEDOCS:
    sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "aeon"
copyright = "BSD-3-Clause License"
author = "aeon developers"

# The full version, including alpha/beta/rc tags
CURRENT_VERSION = f"v{aeon.__version__}"

# If on readthedocs, and we're building the latest version, update tag to generate
# correct links in notebooks
if ON_READTHEDOCS:
    READTHEDOCS_VERSION = os.environ.get("READTHEDOCS_VERSION")
    if READTHEDOCS_VERSION == "latest":
        CURRENT_VERSION = "main"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",  # link to GitHub source code via linkcode_resolve()
    "numpydoc",
    "nbsphinx",  # integrates example notebooks
    "sphinx_design",
    "sphinx_issues",
    "myst_parser",
]

# Notebook thumbnails
nbsphinx_thumbnails = {
    "examples/02_classification": "examples/img/tsc.png",
}

# Use bootstrap CSS from theme.
panels_add_bootstrap_css = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The main toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    ".ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

add_module_names = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# see http://stackoverflow.com/q/12206334/562769
numpydoc_show_class_members = True
# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

numpydoc_validation_checks = {"all"}

# generate autosummary even if no references
autosummary_generate = True

# Members and inherited-members default to showing methods and attributes from a
# class or those inherited.
# Member-order orders the documentation in the order of how the members are defined in
# the source code.
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# Link to GitHub repo for github_issues extension
issues_github_path = "aeon-toolkit/aeon"

# MyST Parser configuration

# When building HTML using the sphinx.ext.mathjax (enabled by default),
# Myst-Parser injects the tex2jax_ignore (MathJax v2) and mathjax_ignore (MathJax v3)
# classes in to the top-level section of each MyST document, and adds some default
# configuration. This ensures that MathJax processes only math, identified by the
# dollarmath and amsmath extensions, or specified in math directives. We here silence
# the corresponding warning that this override happens.
suppress_warnings = ["myst.mathjax"]

# Recommended by sphinx_design when using the MyST Parser
myst_enable_extensions = ["colon_fence"]

myst_heading_anchors = 2


def linkcode_resolve(domain, info):
    """Return URL to source code corresponding.

    Parameters
    ----------
    domain : str
    info : dict

    Returns
    -------
    url : str
    """

    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/main/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(aeon.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "aeon/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    return "https://github.com/aeon-toolkit/aeon/blob/%s/%s" % (
        github_tag,
        filename,
    )


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML.
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "announcement": """
        <b>DISCLAIMER</b>: This is a fork of the sktime repository. Work is ongoing to change links
        and documentation to reflect this.

        Join our
        <a href="https://join.slack.com/t/scikit-timeworkspace/shared_invite/zt-1pkhua342-W_W24XuAZt2JZU1GniK2YA">Slack</a>
        to discuss the projects goals, ask usage questions and discuss contributions.

        We do not recommend using this repository in any
        production setting, but welcome any contributors willing to help us update the
        project. Links and buttons are likely to be broken in the current state.
    """,  # noqa: E501
    "sidebar_hide_name": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/aeon-toolkit/aeon/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#D71414",
        "color-brand-content": "#EB1414",
    },
    "dark_css_variables": {
        "color-brand-primary": "#FF1414",
        "color-brand-content": "#EB3C3C",
    },
    "footer_icons": [
        {
            "name": "Slack",
            "url": "https://join.slack.com/t/aeon-toolkit/shared_invite/zt-1plkevy4x-vAg1dAUXcuoR38FjY9nxzg",  # noqa: E501
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 1024 1024" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                <path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zM361.5 580.2c0 27.8-22.5 50.4-50.3 50.4-13.3 0-26.1-5.3-35.6-14.8-9.4-9.5-14.7-22.3-14.7-35.6 0-27.8 22.5-50.4 50.3-50.4h50.3v50.4zm134 134.4c0 27.8-22.5 50.4-50.3 50.4-27.8 0-50.3-22.6-50.3-50.4V580.2c0-27.8 22.5-50.4 50.3-50.4 13.3 0 26.1 5.3 35.6 14.8s14.7 22.3 14.7 35.6v134.4zm-50.2-218.4h-134c-27.8 0-50.3-22.6-50.3-50.4 0-27.8 22.5-50.4 50.3-50.4h134c27.8 0 50.3 22.6 50.3 50.4-.1 27.9-22.6 50.4-50.3 50.4zm0-134.4c-13.3 0-26.1-5.3-35.6-14.8S395 324.8 395 311.4c0-27.8 22.5-50.4 50.3-50.4 27.8 0 50.3 22.6 50.3 50.4v50.4h-50.3zm83.7-50.4c0-27.8 22.5-50.4 50.3-50.4 27.8 0 50.3 22.6 50.3 50.4v134.4c0 27.8-22.5 50.4-50.3 50.4-27.8 0-50.3-22.6-50.3-50.4V311.4zM579.3 765c-27.8 0-50.3-22.6-50.3-50.4v-50.4h50.3c27.8 0 50.3 22.6 50.3 50.4 0 27.8-22.5 50.4-50.3 50.4zm134-134.4h-134c-13.3 0-26.1-5.3-35.6-14.8S529 593.6 529 580.2c0-27.8 22.5-50.4 50.3-50.4h134c27.8 0 50.3 22.6 50.3 50.4 0 27.8-22.5 50.4-50.3 50.4zm0-134.4H663v-50.4c0-27.8 22.5-50.4 50.3-50.4s50.3 22.6 50.3 50.4c0 27.8-22.5 50.4-50.3 50.4z"></path>
            </svg>
            """,  # noqa: E501
            "class": "",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/company/aeon-toolkit/",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 1024 1024" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                <path d="M880 112H144c-17.7 0-32 14.3-32 32v736c0 17.7 14.3 32 32 32h736c17.7 0 32-14.3 32-32V144c0-17.7-14.3-32-32-32zM349.3 793.7H230.6V411.9h118.7v381.8zm-59.3-434a68.8 68.8 0 1 1 68.8-68.8c-.1 38-30.9 68.8-68.8 68.8zm503.7 434H675.1V608c0-44.3-.8-101.2-61.7-101.2-61.7 0-71.2 48.2-71.2 98v188.9H423.7V411.9h113.8v52.2h1.6c15.8-30 54.5-61.7 112.3-61.7 120.2 0 142.3 79.1 142.3 181.9v209.4z"></path>
            </svg>
            """,  # noqa: E501
            "class": "",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/aeon_toolbox",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 1024 1024" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                <path d="M928 254.3c-30.6 13.2-63.9 22.7-98.2 26.4a170.1 170.1 0 0 0 75-94 336.64 336.64 0 0 1-108.2 41.2A170.1 170.1 0 0 0 672 174c-94.5 0-170.5 76.6-170.5 170.6 0 13.2 1.6 26.4 4.2 39.1-141.5-7.4-267.7-75-351.6-178.5a169.32 169.32 0 0 0-23.2 86.1c0 59.2 30.1 111.4 76 142.1a172 172 0 0 1-77.1-21.7v2.1c0 82.9 58.6 151.6 136.7 167.4a180.6 180.6 0 0 1-44.9 5.8c-11.1 0-21.6-1.1-32.2-2.6C211 652 273.9 701.1 348.8 702.7c-58.6 45.9-132 72.9-211.7 72.9-14.3 0-27.5-.5-41.2-2.1C171.5 822 261.2 850 357.8 850 671.4 850 843 590.2 843 364.7c0-7.4 0-14.8-.5-22.2 33.2-24.3 62.3-54.4 85.5-88.2z"></path>
            </svg>
            """,  # noqa: E501
            "class": "",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/aeon-toolkit/aeon",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 1024 1024" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                <path d="M511.6 76.3C264.3 76.2 64 276.4 64 523.5 64 718.9 189.3 885 363.8 946c23.5 5.9 19.9-10.8 19.9-22.2v-77.5c-135.7 15.9-141.2-73.9-150.3-88.9C215 726 171.5 718 184.5 703c30.9-15.9 62.4 4 98.9 57.9 26.4 39.1 77.9 32.5 104 26 5.7-23.5 17.9-44.5 34.7-60.8-140.6-25.2-199.2-111-199.2-213 0-49.5 16.3-95 48.3-131.7-20.4-60.5 1.9-112.3 4.9-120 58.1-5.2 118.5 41.6 123.2 45.3 33-8.9 70.7-13.6 112.9-13.6 42.4 0 80.2 4.9 113.5 13.9 11.3-8.6 67.3-48.8 121.3-43.9 2.9 7.7 24.7 58.3 5.5 118 32.4 36.8 48.9 82.7 48.9 132.3 0 102.2-59 188.1-200 212.9a127.5 127.5 0 0 1 38.1 91v112.5c.8 9 0 17.9 15 17.9 177.1-59.7 304.6-227 304.6-424.1 0-247.2-200.4-447.3-447.5-447.3z"></path>
            </svg>
            """,  # noqa: E501
            "class": "",
        },
        {
            "name": "ReadTheDocs",
            "url": "https://readthedocs.org/projects/aeon-toolkit/",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" role="img" viewBox="0 0 24 24" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg">
                <path d="M7.732 0a59.316 59.316 0 0 0-4.977.218V24a62.933 62.933 0 0 1 3.619-.687c.17-.028.34-.053.509-.078.215-.033.43-.066.644-.096l.205-.03zm1.18.003V22.96a61.042 61.042 0 0 1 12.333-.213V1.485A60.859 60.859 0 0 0 8.912.003zm1.707 1.81a.59.59 0 0 1 .015 0c3.06.088 6.125.404 9.167.95a.59.59 0 0 1 .476.686.59.59 0 0 1-.569.484.59.59 0 0 1-.116-.009 60.622 60.622 0 0 0-8.992-.931.59.59 0 0 1-.573-.607.59.59 0 0 1 .592-.572zm-4.212.028a.59.59 0 0 1 .578.565.59.59 0 0 1-.564.614 59.74 59.74 0 0 0-2.355.144.59.59 0 0 1-.04.002.59.59 0 0 1-.595-.542.59.59 0 0 1 .54-.635c.8-.065 1.6-.114 2.401-.148a.59.59 0 0 1 .035 0zm4.202 2.834a.59.59 0 0 1 .015 0 61.6 61.6 0 0 1 9.167.8.59.59 0 0 1 .488.677.59.59 0 0 1-.602.494.59.59 0 0 1-.076-.006 60.376 60.376 0 0 0-8.99-.786.59.59 0 0 1-.584-.596.59.59 0 0 1 .582-.583zm-4.211.097a.59.59 0 0 1 .587.555.59.59 0 0 1-.554.622c-.786.046-1.572.107-2.356.184a.59.59 0 0 1-.04.003.59.59 0 0 1-.603-.533.59.59 0 0 1 .53-.644c.8-.078 1.599-.14 2.4-.187a.59.59 0 0 1 .036 0zM10.6 7.535a.59.59 0 0 1 .015 0c3.06-.013 6.125.204 9.167.65a.59.59 0 0 1 .498.67.59.59 0 0 1-.593.504.59.59 0 0 1-.076-.006 60.142 60.142 0 0 0-8.992-.638.59.59 0 0 1-.592-.588.59.59 0 0 1 .573-.592zm1.153 2.846a61.093 61.093 0 0 1 8.02.515.59.59 0 0 1 .509.66.59.59 0 0 1-.586.514.59.59 0 0 1-.076-.005 59.982 59.982 0 0 0-8.99-.492.59.59 0 0 1-.603-.577.59.59 0 0 1 .578-.603c.382-.008.765-.012 1.148-.012zm1.139 2.832a60.92 60.92 0 0 1 6.871.394.59.59 0 0 1 .52.652.59.59 0 0 1-.577.523.59.59 0 0 1-.076-.004 59.936 59.936 0 0 0-8.991-.344.59.59 0 0 1-.61-.568.59.59 0 0 1 .567-.611c.765-.028 1.53-.042 2.296-.042z"></path>
            </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
}

html_logo = "images/aeon-logo-horizontal.png"
html_favicon = "images/aeon-favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = [
    "js/dynamic_table.js",
]

html_show_sourcelink = False

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "aeondoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
    # Latex figure (float) alignment
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "aeon.tex", "aeon Documentation", "aeon developers", "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "aeon", "aeon Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "aeon",
        "aeon Documentation",
        author,
        "aeon",
        "One line description of project.",
        "Miscellaneous",
    ),
]


def _make_estimator_overview(app):
    """Make estimator overview table."""
    import pandas as pd

    from aeon.registry import all_estimators

    def _process_author_info(author_info):
        """
        Process author information from source code files.

        Parameters
        ----------
        author_info : str
            Author information string from source code files.

        Returns
        -------
        author_info : str
            Preprocessed author information.

        Notes
        -----
        A list of author names is turned into a string.
        Multiple author names will be separated by a comma,
        with the final name always preceded by "&".
        """
        if isinstance(author_info, list):
            if len(author_info) > 1:
                return ", ".join(author_info[:-1]) + " & " + author_info[-1]
            else:
                return author_info[0]
        else:
            return author_info

    def _does_not_start_with_underscore(input_string):
        return not input_string.startswith("_")

    # creates dataframe as df
    COLNAMES = ["Class Name", "Estimator Type", "Authors"]

    df = pd.DataFrame([], columns=COLNAMES)

    for modname, modclass in all_estimators():
        algorithm_type = "::".join(str(modclass).split(".")[1:-2])
        try:
            author_info = _process_author_info(modclass.__author__)
        except AttributeError:
            try:
                author_info = _process_author_info(
                    import_module(modclass.__module__).__author__
                )
            except AttributeError:
                author_info = "no author info"

        # includes part of class string
        modpath = str(modclass)[8:-2]
        path_parts = modpath.split(".")
        # joins strings excluding starting with '_'
        clean_path = ".".join(list(filter(_does_not_start_with_underscore, path_parts)))
        # adds html link reference
        modname = str(
            '<a href="https://www.aeon-toolkit.org/en/latest/api_reference'
            + "/auto_generated/"
            + clean_path
            + '.html">'
            + modname
            + "</a>"
        )

        record = pd.DataFrame([modname, algorithm_type, author_info], index=COLNAMES).T
        df = pd.concat([df, record], ignore_index=True)
    with open("estimator_overview_table.md", "w") as file:
        df.to_markdown(file, index=False)


def setup(app):
    """Set up sphinx builder.

    Parameters
    ----------
    app : Sphinx application object
    """

    def adds(pth):
        print("Adding stylesheet: %s" % pth)  # noqa: T201, T001
        app.add_css_file(pth)

    adds("fields.css")  # for parameters, etc.

    app.connect("builder-inited", _make_estimator_overview)


# -- Extension configuration -------------------------------------------------

# -- Options for nbsphinx extension ---------------------------------------
nbsphinx_execute = "never"  # always  # whether to run notebooks
nbsphinx_allow_errors = False  # False
nbsphinx_timeout = 600  # seconds, set to -1 to disable timeout

# add Binder launch buttom at the top
current_file = "{{ env.doc2path( env.docname, base=None) }}"

# make sure Binder points to latest stable release, not main
binder_url = f"https://mybinder.org/v2/gh/aeon-toolkit/aeon/{github_tag}?filepath={current_file}"  # noqa
nbsphinx_prolog = f"""
.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: {binder_url}

|Binder|_
"""

# add link to original notebook at the bottom
notebook_url = f"https://github.com/aeon-toolkit/aeon/tree/{github_tag}/{current_file}"
nbsphinx_epilog = f"""
----

Generated using nbsphinx_. The Jupyter notebook can be found here_.

.. _here: {notebook_url}
.. _nbsphinx: https://nbsphinx.readthedocs.io/
"""

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
}

# -- Options for _todo extension ----------------------------------------------
todo_include_todos = False
