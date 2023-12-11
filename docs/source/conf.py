project = "spotter"
copyright = "2023, Lionel Garcia, Benjamin Rackham"
author = "Lionel Garcia, Benjamin Rackham"
release = "0.0.2"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
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
html_short_title = "spotter"
html_title = f"{html_short_title}"

html_css_files = ["style.css"]
myst_url_schemes = ("http", "https")
