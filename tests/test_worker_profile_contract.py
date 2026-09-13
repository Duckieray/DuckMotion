from __future__ import annotations

import inspect

import ltx_convrot_worker
from model_recipes import LTX25_CONVROT_TWO_STAGE_AV


def test_convrot_worker_is_bound_to_registered_profile_id():
    assert ltx_convrot_worker.EXECUTION_PROFILE_ID == LTX25_CONVROT_TWO_STAGE_AV.profile_id
    source = inspect.getsource(ltx_convrot_worker._run)
    assert "profile_id != EXECUTION_PROFILE_ID" in source
    assert '"execution_profile": EXECUTION_PROFILE_ID' in source


def test_worker_metadata_uses_profile_identity_not_model_brand():
    source = inspect.getsource(ltx_convrot_worker)
    assert '"sampling": "redgraft_convrot_two_stage"' not in source
    assert '"sampling": "ltx25_convrot_two_stage_av"' in source
