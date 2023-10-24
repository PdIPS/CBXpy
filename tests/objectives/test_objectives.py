import cbx.objectives as cob
import pytest
import numpy as np

class test_abstract_objective:
    @pytest.fixture
    def f(self):
        raise NotImplementedError()

    def test_eval(self, f):
        x = np.random.uniform(-1,1,(6,5,7))
        z = f(x)
        assert z.shape == (6,5)

class Test_Rastrigin(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Rastrigin()
    
class Test_Rastrigin_multimodal(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Rastrigin_multimodal()
    
class Test_Ackley(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Ackley()
    
class Test_Ackley_multimodal(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Ackley_multimodal()
    
class Test_three_hump_camel(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.three_hump_camel()
    
class Test_McCormick(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.McCormick()
    
class Test_Rosenbrock(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Rosenbrock()
    
class Test_accelerated_sinus(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.accelerated_sinus()
    
class Test_nd_sinus(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.nd_sinus()
    
class Test_p_4th_order(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.p_4th_order()
    
class Test_Quadratic(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Quadratic()
    
class Test_Banana(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Banana()
    
class Test_Bimodal(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Bimodal()
    
class Test_Unimodal(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Unimodal()
    
class Test_Bukin6(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Bukin6()
    
class Test_cross_in_tray(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.cross_in_tray()
    
class Test_Easom(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Easom()
    
class Test_drop_wave(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.drop_wave()
    
class Test_Holder_table(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.Holder_table()
    
class Test_snowflake(test_abstract_objective):
    @pytest.fixture
    def f(self):
        return cob.snowflake()