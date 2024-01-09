# mypy: ignore-errors

import nox

ALL_PYTHON_VS = ["3.9", "3.10", "3.11"]


@nox.session(python=ALL_PYTHON_VS)
def test(session):
    session.install(".[test,test-math]")
    session.run("pytest", "-n", "auto", *session.posargs)


@nox.session(python=["3.9"])
def comparison(session):
    session.install(".[test,comparison]", "numpy<1.22")
    session.run("python", "-c", "import starry")
    session.run("python", "-c", "import theano")
    if session.posargs:
        args = session.posargs
    else:
        args = ("tests/experimental/starry",)
    session.run(
        "pytest",
        "-n",
        "auto",
        *args,
        env={"JAX_ENABLE_X64": "True"},
    )
