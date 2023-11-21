import cbx
import pytest
from cbx.plotting import plot_dynamic, plot_dynamic_history

class test_plot_dynamic:
    @pytest.fixture
    def f(self):
        return cbx.objectives.Quadratic()

class Test_plot_dynamic(test_plot_dynamic):
    @pytest.fixture
    def plot(self):
        return plot_dynamic
    
    def test_plot_init(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2)
        dyn.step()
        plotter = plot(dyn)
        plotter.init_plot()

    def test_plot_x(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2)
        dyn.step()
        plotter = plot(dyn)
        plotter.init_plot()
        plotter.update()

class Test_plot_dynamic_history(test_plot_dynamic):
    @pytest.fixture
    def plot(self):
        return plot_dynamic_history
    
    def test_max_it(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x'], max_it=1)
        dyn.step()
        plotter = plot(dyn)
        assert plotter.max_it == 1

    def test_plot_at_ind_x(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x'])
        dyn.step()
        plotter = plot(dyn)
        plotter.plot_at_ind(0)

    def test_plot_at_ind_c(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x', 'consensus'])
        dyn.step()
        plotter = plot(dyn, plot_consensus=True)
        plotter.plot_at_ind(0)

    def test_plot_at_ind_d(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x', 'drift'])
        dyn.step()
        plotter = plot(dyn, plot_drift=True)
        plotter.plot_at_ind(0)

    def test_run_plots(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_list=['x'], max_it=1)
        dyn.step()
        plotter = plot(dyn)
        plotter.run_plots()