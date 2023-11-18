import cbx
import pytest
from cbx.plotting import plot_dynamic


class Test__plotting:
    @pytest.fixture
    def f(self):
        return cbx.objectives.Quadratic()

    def test_max_it(self, f):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x'], max_it=1)
        dyn.step()
        plotter = plot_dynamic(dyn)
        assert plotter.max_it == 1

    def test_plot_at_ind_x(self, f):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x'])
        dyn.step()
        plotter = plot_dynamic(dyn)
        plotter.plot_at_ind(0)

    def test_plot_at_ind_c(self, f):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x', 'consensus'])
        dyn.step()
        plotter = plot_dynamic(dyn, plot_consensus=True)
        plotter.plot_at_ind(0)

    def test_plot_at_ind_d(self, f):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x', 'drift'])
        dyn.step()
        plotter = plot_dynamic(dyn, plot_drift=True)
        plotter.plot_at_ind(0)

    def test_run_plots(self, f):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x'], max_it=1)
        dyn.step()
        plotter = plot_dynamic(dyn)
        plotter.run_plots()