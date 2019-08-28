from testutils import assert_raises

assert pow(3,2) == 9
assert pow(5,3, 100) == 25

#causes overflow
# assert pow(41, 7, 2) == 1
assert pow(7, 2, 49) == 0

def _assert_print(f, args):
    raised = True
    try:
        f()
        raised = False
    finally:
        if raised:
            print('Assertion Failure:', *args)

def _typed(obj):
    return '{}({})'.format(type(obj), obj)


def assert_equal(a, b):
    _assert_print(lambda: a == b, [_typed(a), '==', _typed(b)])


def assert_almost_equal(a, b):
    _assert_print(lambda: a == b, [_typed(a), '==', _typed(b)])


def assert_true(e):
    _assert_print(lambda: e is True, [_typed(e), 'is True'])


def assert_false(e):
    _assert_print(lambda: e is False, [_typed(e), 'is False'])


def assert_isinstance(obj, klass):
    _assert_print(lambda: isinstance(obj, klass), ['isisntance(', _typed(obj), ',', klass, ')'])

def assert_in(a, b):
    _assert_print(lambda: a in b, [a, 'in', b])

def powtest(type):
    if type != float:
        for i in range(-1000, 1000):
            assert_equal(pow(type(i), 0), 1)
            assert_equal(pow(type(i), 1), type(i))
            assert_equal(pow(type(0), 1), type(0))
            assert_equal(pow(type(1), 1), type(1))

        for i in range(-100, 100):
            assert_equal(pow(type(i), 3), i*i*i)

        pow2 = 1
        for i in range(0, 31):
            assert_equal(pow(2, i), pow2)
            if i != 30 : pow2 = pow2*2

        for othertype in (int,):
            for i in list(range(-10, 0)) + list(range(1, 10)):
                ii = type(i)
                for j in range(1, 11):
                    jj = -othertype(j)
                    pow(ii, jj)

    for othertype in int, float:
        for i in range(1, 100):
            zero = type(0)
            exp = -othertype(i/10.0)
            if exp == 0:
                continue
            assert_raises(ZeroDivisionError, pow, zero, exp)

    il, ih = -20, 20
    jl, jh = -5,   5
    kl, kh = -10, 10
    asseq = assert_equal
    if type == float:
        il = 1
        asseq = assert_almost_equal
    elif type == int:
        jl = 0
    elif type == int:
        jl, jh = 0, 15
    for i in range(il, ih+1):
        for j in range(jl, jh+1):
            for k in range(kl, kh+1):
                if k != 0:
                    if type == float or j < 0:
                        assert_raises(TypeError, pow, type(i), j, k)
                        continue
                    asseq(
                        pow(type(i),j,k),
                        pow(type(i),j)% type(k)
                    )


powtest(int)
powtest(float)
