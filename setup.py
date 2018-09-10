""" Setup script for DevBG blog post package. """

from setuptools import setup

setup(
    name = "models",
    author = "Todor Davchev",
    description = ("LSTMs as a deep approach to predicting recent stochastic futures"
                   "Blog post series for Dev.bg, Bulgaria"),
    url = "https://github.com/yadrimz/stochastic_futures_prediction",
    packages=['models', 'utils']
)