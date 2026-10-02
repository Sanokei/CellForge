import random
import pytest
from scripts.build_arithmetic_specialist import build_arithmetic
from cellforge.ir.graph import GraphError, Limits
from cellforge.runtime.arithmetic import DecimalAdditionPredictor


def test_all_digit_carries_and_length_transfer():
    graph = build_arithmetic()
    graph.validate(Limits(strict_microcells=True))
    predict = DecimalAdditionPredictor(graph)
    rng = random.Random(83751)
    for length in (1,2,7,31,129,513):
        for _ in range(4):
            a = ''.join(str(rng.randrange(10)) for _ in range(length))
            b = ''.join(str(rng.randrange(10)) for _ in range(rng.randrange(1,length+1)))
            assert predict(a,b)['sum'] == str(int(a)+int(b))
    for a in range(20):
        for b in range(20):
            assert predict(str(a),str(b))['sum'] == str(a+b)
    assert predict('9'*4096,'1')['sum'] == '1'+'0'*4096
    assert predict('000','000')['sum'] == '0'


@pytest.mark.parametrize('a,b',[('-1','2'),('1.2','3'),('','0'),('１２','3'),(1,'3'),('9'*4097,'1')])
def test_invalid_inputs(a,b):
    with pytest.raises(GraphError):
        DecimalAdditionPredictor(build_arithmetic())(a,b)
