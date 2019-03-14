"""
Unit and regression test for the responsefun package.
"""

# Import package, test suite, and other packages as needed
import responsefun
import pytest
import sys

def test_responsefun_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "responsefun" in sys.modules
