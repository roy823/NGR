import numpy as np
import pytest
import torch

from ngr import register_correspondences, register_with_surfaces
from ngr import registration as api
from ngr import _surface as surface
from ngr._core import bank


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def points():
    p = np.random.default_rng(7).normal(size=(48, 3))
    angle = .4
    R = np.array([[np.cos(angle), -np.sin(angle), 0.],
                  [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    t = np.array([.2, -.1, .3])
    return p, p @ R.T + t, R, t


@pytest.mark.parametrize("radius", [.04, .1, .6])
def test_correspondence_radius_and_output_units(radius):
    p, q, R, t = points()
    unit = radius / .04
    p, q, t = p * unit, q * unit, t * unit
    p0, q0 = p.copy(), q.copy()
    pose = register_correspondences(p, q, inlier_radius=radius)
    np.testing.assert_allclose(pose["R"], R, atol=1e-7)
    np.testing.assert_allclose(pose["t"], t, atol=1e-7)
    np.testing.assert_array_equal(p, p0)
    np.testing.assert_array_equal(q, q0)


@pytest.mark.parametrize("radius,budget", [(.1, 16), (.1, 32), (.6, 16), (.6, 32)])
def test_surface_interface_units_and_budgets(radius, budget):
    pytest.importorskip("open3d")
    p, q, R, t = points()
    unit = radius / .1
    p, q, t = p * unit, q * unit, t * unit
    p0, q0 = p.copy(), q.copy()
    pose = register_with_surfaces(p, q, p, q, inlier_radius=radius, budget=budget)
    np.testing.assert_allclose(pose["R"], R, atol=1e-6)
    np.testing.assert_allclose(pose["t"], t, atol=1e-6)
    np.testing.assert_array_equal(p, p0)
    np.testing.assert_array_equal(q, q0)


def test_pair_branch_works_with_surfaces():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(19)
    p = 20 * rng.normal(size=(48, 3))
    t = np.array([.2, -.1, .3])
    q = 20 * rng.normal(size=(48, 3))
    q[:8] = p[:8] + t
    candidates = bank(p * .4, q * .4)
    assert candidates["seeds"].shape[1] == 2
    pose = register_with_surfaces(p, q, p, p + t, budget=16)
    np.testing.assert_allclose(pose["R"], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(pose["t"], t, atol=1e-6)


@pytest.mark.parametrize("radius", [.1, .6])
def test_surface_preparation_keeps_first_observations_in_order(radius):
    values = radius * np.array([[.01, 0., 0.], [.02, 0., 0.],
                               [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                               [1., 1., 0.], [1., 0., 1.]])
    scale = .04 / radius
    expected = (values * scale)[[0, 2, 3, 4, 5, 6]] / scale
    np.testing.assert_array_equal(api._prepare_surface(values, radius), expected)


@pytest.mark.parametrize("order", [2, 3])
def test_direct_pose_role_and_translation_scaling(monkeypatch, order):
    p, q, _, _ = points()
    raw_R = np.eye(3)[None]
    raw_t = np.array([[.03, -.02, .01]])
    monkeypatch.setattr(api, "bank", lambda *args: dict(
        rotation=raw_R, translation=raw_t, scores=np.array([1.]),
        seeds=np.arange(order)[None]))
    calls = []
    def direct(*args):
        calls.append(args)
        return raw_R.copy(), raw_t * 2
    monkeypatch.setattr(api, "seed_poses", direct)
    candidates = api._candidates(p, q, .6)
    expected = (raw_t / (.04 / .6)) * (.1 / .6)
    np.testing.assert_array_equal(candidates["t"], expected)
    np.testing.assert_array_equal(candidates["native"]["t"], expected[0])
    assert len(candidates["R"]) == 1
    assert len(candidates["aux_R"]) == (1 if order == 3 else 0)
    assert len(calls) == (1 if order == 3 else 0)
    if order == 3:
        np.testing.assert_array_equal(candidates["aux_t"],
                                      ((raw_t * 2) / (.04 / .6)) * (.1 / .6))


def test_global_budget_uses_nested_extra_queue(monkeypatch):
    class Evidence:
        def __init__(self, source, target):
            pass
        def score(self, R, t, information=False, early=False):
            return 1 + t[:, 0]
    class ICP:
        def __init__(self, source, target):
            pass
        def apply(self, pose):
            return pose, [{}, {}]
    monkeypatch.setattr(surface, "Evidence", Evidence)
    monkeypatch.setattr(surface, "SurfaceICP", ICP)
    monkeypatch.setattr(surface, "mae", lambda p, q, R, t: 100 - t[:, 0])
    p, q, _, _ = points()
    R, t = np.eye(3)[None], np.zeros((1, 3))
    aux_R = np.repeat(R, 20, axis=0)
    aux_t = np.c_[.11 * np.arange(1, 21), np.zeros((20, 2))]
    native = dict(valid=True, R=R[0], t=t[0])
    small = surface.process(p, q, p, q, R, t, native, aux_R, aux_t, maximum=16)
    large = surface.process(p, q, p, q, R, t, native, aux_R, aux_t, maximum=32)
    assert small["retained_calls"] == large["retained_calls"] == 1
    assert small["total_calls"] == small["fresh_ICP_calls"] == 16
    assert large["total_calls"] == large["fresh_ICP_calls"] == 17
    np.testing.assert_array_equal(small["bank"]["extra_initial_t"],
                                  large["bank"]["extra_initial_t"][:15])
    np.testing.assert_array_equal(small["bank"]["refined_t"],
                                  large["bank"]["refined_t"][:16])
    assert small["indices"]["16"] == large["indices"]["16"]


def test_invalid_primary_pool_keeps_invalid_output():
    post = dict(valid=np.array([False, True]), refined_M=np.ones(2),
                refined_G=np.ones(2))
    assert surface.choose(post, np.ones(2), keep=1, budget=32) is None


@pytest.mark.parametrize("p,q", [(np.zeros((2, 3)), np.zeros((2, 3))),
    (np.zeros((4, 3)), np.zeros((5, 3))), (np.full((4, 3), np.nan), np.zeros((4, 3)))])
def test_invalid_correspondences_are_rejected(p, q):
    with pytest.raises(ValueError):
        register_correspondences(p, q)


@pytest.mark.parametrize("radius", [0, -1, np.nan, np.inf])
def test_invalid_physical_radius_is_rejected(radius):
    p, q, _, _ = points()
    with pytest.raises(ValueError):
        register_correspondences(p, q, inlier_radius=radius)


@pytest.mark.parametrize("budget", [0, 15, 17, 33, 16., 32., "16", None])
def test_invalid_surface_budget_is_rejected(budget):
    p, q, _, _ = points()
    with pytest.raises(ValueError):
        register_with_surfaces(p, q, p, q, budget=budget)


def test_surfaces_must_retain_six_observations():
    p, q, _, _ = points()
    with pytest.raises(ValueError, match="six retained"):
        register_with_surfaces(p, q, np.zeros((20, 3)), q)
