import sys
import pkg_resources

#simple package checker and version used for the dysertation

print(f"Python version: {sys.version}")

packages = [
    'pandas', 'numpy', 'scikit-learn', 'torch',
    'beautifulsoup4', 'requests', 'warcio',
    'python-whois', 'tldextract', 'statsmodels',
    'matplotlib', 'seaborn', 'joblib', 'openpyxl',
    'pyarrow', 'lxml'
]

for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: NOT FOUND")