import pytest

@pytest.fixture(scope="session")
def tmpfile(tmp_path_factory, content=""):
    fn = tmp_path_factory.mktemp("pytest_data") / "test.tmpfile"
    fn.write_text(content)
    return fn.as_posix()
