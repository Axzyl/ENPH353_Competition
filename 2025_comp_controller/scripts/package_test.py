#!/usr/bin/env python3

import sys
print("Python:", sys.version)
print("Path:", sys.path)

try:
    import pytesseract
    print("pytesseract OK:", pytesseract.__file__)
except ImportError as e:
    print("pytesseract FAILED:", e)