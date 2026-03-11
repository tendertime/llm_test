import sys
print('Python version:', sys.version)

try:
    import datasets
    print('datasets version:', datasets.__version__)
except ImportError:
    print('datasets not installed')
