from setuptools import setup, find_packages
import sys
import pathlib

readme_file = "README.md"
try:
    from pypandoc import convert

    readme = convert(readme_file, "rst")
except ImportError:
    print("pypandoc not available. Readme will be in Markdown", file=sys.stderr)
    readme = open(readme_file, "r").read()

with open("LICENSE") as f:
    license = f.read()


setup(
    name="nllblangs",
    version="0.1.0",
    description="Language codes utility for NLLB",
    long_description=readme,
    author="Onur Çelebi",
    author_email="celebio@fb.com",
    url="https://nolanguageleftbehind.com",
    license=license,
    packages=find_packages("src", exclude=("tests", "docs")) + ["nllblangs.data"],
    package_dir={"nllblangs": "src/nllblangs", "nllblangs.data": "data"},
    include_package_data=True,
    data_files=[("data", ["data/classification_200.tsv", "data/general.tsv"])],
    entry_points={
        "console_scripts": ["nllblangs = nllblangs.cli.__init__:main"],
    },
    install_requires=["pandas", "pypandoc", "openpyxl"],
)
