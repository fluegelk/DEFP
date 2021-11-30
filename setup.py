import setuptools
import pathlib


readme = pathlib.Path(__file__).parent / "README.md"

setuptools.setup(
      name="feedforward",
      version="0.0.1",
      description='Feedforward only training of neural networks.',
      long_description=readme.read_text(encoding='utf-8'),
      long_description_content_type="text/markdown",
      packages=['feedforward'],
      install_requires=['torch', 'torchvision', 'numpy', 'tqdm', 'scikit-learn', 'matplotlib', 'pandas', 'plotly',
                        'seaborn'],
      extras_require={'test': ['pytest']})
