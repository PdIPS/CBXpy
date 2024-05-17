import cbx
import pytest
from cbx.plotting import PlotDynamic, PlotDynamicHistory
import matplotlib
matplotlib.use('Agg')

class test_plot_dynamic:
    @pytest.fixture
    def f(self):
        return cbx.objectives.Quadratic()

class Test_plot_dynamic(test_plot_dynamic):
    @pytest.fixture
    def plot(self):
        return PlotDynamic

        
    def test_plot_init(self, f, plot):
        for d in range(1,5):
            dyn = cbx.dynamics.CBO(lambda x: sum(x[i]**2 for i in range(d)), d=d)
            dyn.step()
            plotter = plot(dyn)
            plotter.init_plot()
        
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_plot_x(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2)
        dyn.step()
        plotter = plot(dyn)
        plotter.init_plot()
        plotter.update()

class Test_plot_dynamic_history(test_plot_dynamic):
    @pytest.fixture
    def plot(self):
        return PlotDynamicHistory
    
    def test_max_it(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_args={'names':['x']}, max_it=1)
        dyn.step()
        plotter = plot(dyn)
        assert plotter.max_it == 1

    def test_plot_at_ind_x(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_args={'names':['x']})
        dyn.step()
        plotter = plot(dyn)
        plotter.plot_at_ind(0)

    def test_plot_at_ind_c(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_args={'names':['x', 'consensus']})
        dyn.step()
        plotter = plot(dyn, plot_consensus=True)
        plotter.plot_at_ind(0)

    def test_plot_at_ind_d(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_args={'names':['x', 'drift']})
        dyn.step()
        plotter = plot(dyn, plot_drift=True)
        plotter.plot_at_ind(0)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_run_plots(self, f, plot):
        dyn = cbx.dynamics.CBO(f, d=2, track_args={'names':['x']}, max_it=1)
        dyn.step()
        plotter = plot(dyn)
        plotter.run_plots()