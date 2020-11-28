from deepchall.index import INDEX
import pytest

@pytest.mark.parametrize(
    "lang_name", INDEX["langs"].keys()
)
def test_lang(lang_name):
    lang = INDEX["langs"][lang_name]()
    max_samples = 50
    max_length = None
    lang.init(params={
        "max_samples": max_samples, 
        "max_length": max_length,
    })
    bkd = lang.get()
    gen = bkd.gen(None)
    for _ in range(max_samples):
        sample = next(gen)
        assert bkd.parse(sample)