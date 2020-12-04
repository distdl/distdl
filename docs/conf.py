# -*- coding: utf-8 -*-
from __future__ import unicode_literals

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]

autodoc_member_order = "bysource"

autodoc_mock_imports = ["numpy", "torch", "mpi4py"]

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
version = release = '0.3.1'

pygments_style = 'trac'
# templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/distdl/distdl/issues/%s', '#'),
    'pr': ('https://github.com/distdl/distdl/pull/%s', 'PR #'),
}

import sphinx_py3doc_enhanced_theme  # noqa: E402
html_theme = "sphinx_py3doc_enhanced_theme"
html_theme_path = [sphinx_py3doc_enhanced_theme.get_html_theme_path()]
html_theme_options = {
    'githuburl': 'https://github.com/distdl/distdl/'
}

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


# monkey-patch napoleon to better handle optional parameters in numpydoc
# docstrings; see https://github.com/sphinx-doc/sphinx/issues/6861

def _fixup_napoleon_numpydoc():
    from sphinx.locale import _
    from sphinx.ext.napoleon import NumpyDocstring

    def _process_optional_params(self, fields):
        """
        Split a fields list into separate lists of positional parameters and
        keyword parameters.

        Possibly moves some fields out of their original documented order,
        though in practice, in most cases, optional/keyword parameters should
        always be listed after positional parameters.

        For Numpydoc, a parameter is treated as a keyword parameter if its type
        list ends with the keyword "optional".  In this case, the "optional" is
        removed from its type list, and instead the text "(optional)" is
        prepended to the field description.
        """

        positional = []
        keyword = []

        for name, type_, desc in fields:
            types = [t.strip() for t in type_.split(',')]
            optional = types and types[-1].lower() == 'optional'
            if optional:
                type_ = ', '.join(types[:-1])

                if not desc:
                    desc = ['']
                desc[0] = ('*(optional)* – ' + desc[0]).rstrip()

            if optional or name.startswith(r'\*\*'):
                keyword.append((name, type_, desc))
            else:
                positional.append((name, type_, desc))

        return positional, keyword

    def _parse_parameters_section(self, section):
        fields = self._consume_fields()
        pos_fields, kw_fields = self._process_optional_params(fields)
        if self._config.napoleon_use_param:
            lines = self._format_docutils_params(pos_fields)
        else:
            lines = self._format_fields(_('Parameters'), pos_fields)

        if self._config.napoleon_use_keyword:
            if self._config.napoleon_use_param:
                lines = lines[:-1]
            lines.extend(self._format_docutils_params(
                kw_fields, field_role='keyword', type_role='kwtype'))
        else:
            lines.extend(self._format_fields(
                _('Keyword Arguments'), kw_fields))

        return lines

    def _parse_other_parameters_section(self, section):
        fields = self._consume_fields()
        pos_fields, kw_fields = self._process_optional_params(fields)
        return self.format_fields(
                _('Other Parameters'), pos_fields + kw_fields)

    NumpyDocstring._process_optional_params = _process_optional_params
    NumpyDocstring._parse_parameters_section = _parse_parameters_section
    NumpyDocstring._parse_other_parameters_section = _parse_other_parameters_section


_fixup_napoleon_numpydoc()
