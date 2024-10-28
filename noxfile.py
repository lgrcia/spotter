# mypy: ignore-errors

import nox

ALL_PYTHON_VS = ["3.10", "3.11"]


@nox.session(python=ALL_PYTHON_VS)
def test(session):
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session(python=["3.10"])
def comparison(session):
    session.install(".[test,comparison]")
    if session.posargs:
        args = session.posargs
    else:
        args = ("tests/starry_comparison",)
    session.run(
        "pytest",
        "-n",
        "auto",
        *args,
        env={"JAX_ENABLE_X64": "True"},
    )
