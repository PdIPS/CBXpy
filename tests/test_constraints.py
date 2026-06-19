import numpy as np
from cbx.constraints import (
    MultiConstraint, NoConstraint, sphereConstraint,
    planeConstraint
)


class TestSphereConstraint:
    def test_zero_level_set_at_radius(self):
        """Constraint is zero exactly on the sphere of radius r."""
        for r in [0.5, 1.0, 2.0, 3.5]:
            c = sphereConstraint(r=r)
            # points exactly on the sphere
            x = np.random.randn(10, 3)
            x = x / np.linalg.norm(x, axis=-1, keepdims=True) * r
            assert np.allclose(c(x), 0.0), f"Failed for r={r}"

    def test_inside_outside_sign(self):
        """Points inside sphere give negative value, outside give positive."""
        c = sphereConstraint(r=2.0)
        x_inside = np.ones((5, 3)) * 0.5   # ||x|| = sqrt(3)*0.5 < 2
        x_outside = np.ones((5, 3)) * 2.0  # ||x|| = sqrt(3)*2 > 2
        assert np.all(c(x_inside) < 0)
        assert np.all(c(x_outside) > 0)

    def test_grad_is_2x(self):
        x = np.random.randn(4, 3)
        c = sphereConstraint(r=1.5)
        assert np.allclose(c.grad(x), 2 * x)

    def test_solve_Id_plus_call_grad_consistent_with_r(self):
        """Result is independent of r when x is on the sphere (c(x)==0)."""
        for r in [1.0, 2.0, 0.5]:
            c = sphereConstraint(r=r)
            x = np.random.randn(5, 3)
            x = x / np.linalg.norm(x, axis=-1, keepdims=True) * r
            x_tilde = np.random.randn(5, 3)
            result = c.solve_Id_plus_call_grad(x, x_tilde)
            # c(x)==0 on the sphere, so factor term vanishes → result == x_tilde
            assert np.allclose(result, x_tilde), f"Failed for r={r}"


class TestNoConstraint:
    def test_call_returns_zeros(self):
        c = NoConstraint()
        x = np.random.randn(3, 5, 4)
        assert np.allclose(c(x), 0.0)
        assert c(x).shape == (3, 5)

    def test_grad_returns_zeros(self):
        c = NoConstraint()
        x = np.random.randn(3, 5, 4)
        assert np.allclose(c.grad(x), 0.0)
        assert c.grad(x).shape == x.shape


class TestPlaneConstraint:
    def test_points_on_plane_are_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = 2.0
        c = planeConstraint(a=a, b=b)
        # x1 = 2 lies on the plane a·x = b
        x = np.zeros((5, 3))
        x[:, 0] = 2.0
        assert np.allclose(c(x), 0.0)

    def test_sign_correct(self):
        a = np.array([1.0, 0.0, 0.0])
        b = 2.0
        c = planeConstraint(a=a, b=b)
        x_above = np.array([[3.0, 0.0, 0.0]])   # a·x = 3 > b
        x_below = np.array([[1.0, 0.0, 0.0]])   # a·x = 1 < b
        assert c(x_above) > 0
        assert c(x_below) < 0


class TestMultiConstraint:
    def test_zero_constraints_returns_x_tilde(self):
        """Empty MultiConstraint is identity for solve_Id_call_times_hessian."""
        mc = MultiConstraint([])
        x = np.random.randn(4, 3)
        x_tilde = np.random.randn(4, 3)
        assert np.allclose(mc.solve_Id_call_times_hessian(x, x_tilde), x_tilde)

    def test_single_constraint_delegates(self):
        """Single-constraint MultiConstraint delegates to the underlying solver."""
        sc = sphereConstraint(r=1.0)
        mc = MultiConstraint([sc])
        x = np.random.randn(4, 3)
        x_tilde = np.random.randn(4, 3)
        assert np.allclose(
            mc.solve_Id_call_times_hessian(x, x_tilde),
            sc.solve_Id_call_times_hessian(x, x_tilde)
        )

    def test_solve_Id_hessian_squared_sum_no_crash_multi(self):
        """Multi-constraint solve_Id_hessian_squared_sum runs without AttributeError."""
        sc1 = sphereConstraint(r=1.0)
        sc2 = sphereConstraint(r=1.0)
        mc = MultiConstraint([sc1, sc2])
        x = np.random.randn(4, 3) * 0.1 + 1.0
        x_tilde = np.random.randn(4, 3)
        result = mc.solve_Id_hessian_squared_sum(x, x_tilde)
        assert result.shape == x_tilde.shape

    def test_call_sums_constraints(self):
        sc1 = sphereConstraint(r=1.0)
        sc2 = NoConstraint()
        mc = MultiConstraint([sc1, sc2])
        x = np.random.randn(5, 3)
        assert np.allclose(mc(x), sc1(x))
