import pytest
from monograph import Config


def test_defaults_are_local_first():
    cfg = Config()
    assert cfg.local_only is True
    assert cfg.narrate_backend == "f5"


def test_require_local_ok_blocks_network_by_default():
    cfg = Config()
    with pytest.raises(PermissionError):
        cfg.require_local_ok("some-cloud-backend")
    cfg.local_only = False
    cfg.require_local_ok("some-cloud-backend")   # no raise once opted in
