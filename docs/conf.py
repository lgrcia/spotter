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
    "repository_url": "https://github.com/lgrcia/prose",
    "use_repository_button": True,
}

nb_render_image_options = {"align": "center"}

myst_enable_extensions = [
    "dollarmath",
]

nb_execution_mode = "off"
html_logo = "_static/spotter.png"
html_css_files = ["style.css"]
myst_url_schemes = ("http", "https")

plot_html_show_formats = False
plot_html_show_source_link = False
