# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sphinx_py3doc_enhanced_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]

templates_path = ['_templates']
autosummary_generate = True
numpydoc_show_class_members = False
autosectionlabel_prefix_document = True

source_suffix = '.rst'
master_doc = 'index'
project = 'DistDL'
year = '2020'
author = 'Russell J. Hewett'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.1.0'

pygments_style = 'trac'
# templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/distdl/distdl/issues/%s', '#'),
    'pr': ('https://github.com/distdl/distdl/pull/%s', 'PR #'),
}


# import sphinx_bootstrap_theme
# html_theme = "bootstrap"
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

html_theme = "sphinx_py3doc_enhanced_theme"
html_theme_path = [sphinx_py3doc_enhanced_theme.get_html_theme_path()]
html_theme_options = {
    'githuburl': 'https://github.com/distdl/distdl/'
}

# import pytorch_sphinx_theme
# html_theme = 'pytorch_sphinx_theme'
# html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# # Theme options are theme-specific and customize the look and feel of a theme
# # further.  For a list of options available for each theme, see the
# # documentation.

# html_theme_options = {
#     'pytorch_project': 'docs',
#     'canonical_url': 'https://pytorch.org/docs/stable/',
#     'collapse_navigation': False,
#     'display_version': True,
#     'logo_only': True,
# }


html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'PyTorch': ('https://pytorch.org/docs/master/', None),
    'mpi4py': ('https://mpi4py.readthedocs.io/en/stable/',
               'https://mpi4py.readthedocs.io/en/stable/objects.inv'),
}
