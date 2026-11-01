import numpy as np
import pytest
import torch

from ngr import prepare_indoor_surface, register_correspondences, register_indoor


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def points():
    p = np.random.default_rng(7).normal(size=(48, 3))
    angle = .4
    R = np.array([[np.cos(angle), -np.sin(angle), 0.],
                  [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    t = np.array([.2, -.1, .3])
    return p, p @ R.T + t, R, t


def test_correspondences_recover_rigid_motion_and_preserve_inputs():
    p, q, R, t = points()
    p0, q0 = p.copy(), q.copy()
    pose = register_correspondences(p, q)
    np.testing.assert_allclose(pose["R"], R, atol=1e-7)
    np.testing.assert_allclose(pose["t"], t, atol=1e-7)
    np.testing.assert_array_equal(p, p0)
    np.testing.assert_array_equal(q, q0)


def test_surface_interface_recovers_rigid_motion():
    p, q, R, t = points()
    pose = register_indoor(p, q, prepare_indoor_surface(p), prepare_indoor_surface(q))
    np.testing.assert_allclose(pose["R"], R, atol=1e-6)
    np.testing.assert_allclose(pose["t"], t, atol=1e-6)


def test_surface_preparation_keeps_observed_representatives_in_order():
    points = np.array([[.001, 0., 0.], [.002, 0., 0.], [.2, 0., 0.]])
    np.testing.assert_array_equal(prepare_indoor_surface(points), points[[0, 2]])


@pytest.mark.parametrize("p,q", [(np.zeros((2, 3)), np.zeros((2, 3))),
    (np.zeros((4, 3)), np.zeros((5, 3))), (np.full((4, 3), np.nan), np.zeros((4, 3)))])
def test_invalid_correspondences_are_rejected(p, q):
    with pytest.raises(ValueError):
        register_correspondences(p, q)
