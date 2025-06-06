project = "spotter"
copyright = "2023 - 2024, Lionel Garcia, Benjamin Rackham"
author = "Lionel Garcia, Benjamin Rackham"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

root_doc = "index"

html_theme_options = {
    "repository_url": "https://github.com/lgrcia/spotter",
    "use_repository_button": True,
}

nb_render_image_options = {"align": "center"}

myst_enable_extensions = [
    "dollarmath",
]

html_logo = "_static/spotter.png"
myst_url_schemes = ("http", "https")

plot_html_show_formats = False
plot_html_show_source_link = False

autoapi_dirs = ["../spotter"]
autoapi_ignore = ["*_version*", "*/types*"]
autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    # "imported-members",
]
# autoapi_add_toctree_entry = False
autoapi_template_dir = "_autoapi_templates"

suppress_warnings = ["autoapi.python_import_resolution"]

nb_execution_excludepatterns = ["flux_gp.ipynb"]
plot_include_source = True
