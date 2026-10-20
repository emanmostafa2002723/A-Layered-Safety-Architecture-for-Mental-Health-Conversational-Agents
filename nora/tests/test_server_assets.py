"""Synthetic transfer tests; no real model data or server access required."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('link_server_assets', Path(__file__).resolve().parents[1]/'scripts/link_server_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def synthetic(tmp_path):
    source = tmp_path/'source'
    root = tmp_path/'new'
    source.mkdir();root.mkdir()
    (source/'weights').write_bytes(b'not a real model')
    value = {'schema_version':'nora-runtime-assets/1.0','groups':{'test':{
        'target_prefix':'models/test/', 'files':[{'name':'weights','bytes':16,
        'sha256':hashlib.sha256(b'not a real model').hexdigest()}]}}}
    return source, root, value


def test_readonly_plan_then_exact_hardlink_no_copy(tmp_path):
    source, root, manifest = synthetic(tmp_path)
    rows = assets.plan(root, manifest, {'test':str(source)})
    assert rows[0]['status'] == 'link_ready' and list(root.iterdir()) == []
    assert assets.apply(root, rows) == 1
    assert (source/'weights').samefile(root/'models/test/weights')
    assert assets.plan(root, manifest, {})[0]['status'] == 'already_present'


def test_mismatch_or_missing_mapping_never_creates_destination(tmp_path):
    source, root, manifest = synthetic(tmp_path)
    assert assets.plan(root, manifest, {})[0]['status'] == 'missing_source_mapping'
    (source/'weights').write_bytes(b'bad model bytes!')
    rows = assets.plan(root, manifest, {'test':str(source)})
    assert rows[0]['status'] == 'hash_mismatch'
    with pytest.raises(ValueError):
        assets.apply(root, rows)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize('target',['../outside/','/outside/','models/../../outside/','C:/outside/','models\\outside/'])
def test_target_path_escape_is_rejected(tmp_path, target):
    source, root, manifest = synthetic(tmp_path)
    manifest['groups']['test']['target_prefix'] = target
    with pytest.raises(ValueError):
        assets.plan(root, manifest, {'test':str(source)})


def test_existing_mismatched_target_is_not_overwritten(tmp_path):
    source, root, manifest = synthetic(tmp_path)
    target = root/'models/test/weights'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'existing bad data')
    rows = assets.plan(root, manifest, {'test':str(source)})
    assert rows[0]['status'] in {'size_mismatch','hash_mismatch'}
    with pytest.raises(ValueError):
        assets.apply(root, rows)
    assert target.read_bytes() == b'existing bad data'


def test_link_failure_rolls_back_only_new_links(tmp_path, monkeypatch):
    source, root, manifest = synthetic(tmp_path)
    (source/'second').write_bytes(b'not a real model')
    manifest['groups']['test']['files'].append({**manifest['groups']['test']['files'][0], 'name':'second'})
    rows = assets.plan(root, manifest, {'test':str(source)})
    actual_link = assets.os.link
    def fail_second(a,b):
        if str(b).endswith('second'):
            raise OSError('Synthetic failure')
        return actual_link(a,b)
    monkeypatch.setattr(assets.os,'link',fail_second)
    with pytest.raises(OSError):
        assets.apply(root, rows)
    assert not (root/'models/test/weights').exists()
    assert (source/'weights').read_bytes() == b'not a real model'
