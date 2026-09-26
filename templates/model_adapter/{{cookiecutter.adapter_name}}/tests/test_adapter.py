from {{cookiecutter.package_name}} import {{cookiecutter.class_name}}
from spacergbbenchmark.models.base import Model


def test_registered_interface():
    assert issubclass({{cookiecutter.class_name}}, Model)
    assert {{cookiecutter.class_name}}.temporal_mode in {"framewise", "causal", "offline"}
