from deepchall.index import INDEX
from deepchall.runner import Runner
import pytest

@pytest.mark.parametrize(
    "lang_name", INDEX["langs"].keys()
)
def test_lang(lang_name):
    lang = INDEX["langs"][lang_name]()
    max_samples = 50
    max_length = None

    lang.init(params=Runner.make_lang_params(
        lang=lang,
        user_params={
            "max_samples": max_samples, 
            "max_length": max_length,
        }
    ))

    bkd = lang.get()
    gen = bkd.gen()
    try:
        for _ in range(max_samples):
            sample = next(gen)
            assert bkd.parse(sample)
    except StopIteration:
        pass