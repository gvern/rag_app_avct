from setuptools import find_packages, setup

requirements = """
llama-index==0.10.8
streamlit
python-dotenv
"""

setup(
    name="advanced_chatbot",
    author="Gustave Vernay",
    description="{description}",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='==3.10.13',
    include_package_data=True,
    scripts=[],
    zip_safe=False,
)
