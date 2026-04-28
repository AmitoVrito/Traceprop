"""Build Cython extensions for traceprop."""

from setuptools import Extension, setup

from Cython.Build import cythonize

extensions = [
    Extension(
        "traceprop._c_ext.graph_ops",
        ["traceprop/_c_ext/graph_ops.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
