# taken from respondo

import os


def update_testdata(session):
    import subprocess

    testdata_dir = os.path.join(os.path.dirname(__file__), "testdata")
    cmd = [testdata_dir + "/0_download_testdata.sh"]
    subprocess.check_call(cmd)


#
# Pytest Hooks
#


def pytest_addoption(parser):
    parser.addoption(
        "--skip-update", default=False, action="store_true", help="Skip updating testdata"
    )


def pytest_collection(session):
    if not session.config.option.skip_update:
        update_testdata(session)
