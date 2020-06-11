# Automatically created by: shub deploy

from setuptools import setup, find_packages

setup(
    name         = 'project',
    version      = '1.0',
    packages     = find_packages(),
    package_data={
        'tutorial': ['resources/all_new_urls_1.json']
    },
    entry_points = {'scrapy': ['settings = tutorial.settings']},
)
