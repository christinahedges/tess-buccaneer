import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

from tess_buccaneer import PACKAGEDIR, BoxTransit, TPFModel, __version__


def test_version():
    assert __version__ == "0.1.0"


def test_tpfmodel():
    tpf = lk.read(f"{PACKAGEDIR}/data/tpf1.fits")
    # tm = TPFModel(tpf)
    # tm.fit()
    # assert(np.isclose(tm.bkg0, 100, rtol=3).all())
    # assert(tm.minframe.shape == tpf.shape[1:])
    # tm.make_pdf()

    duration = 2.4264 / 24
    t0 = 2456258.0621 - 2457000
    period = 3.4252602
    transitmodel = BoxTransit(period=period, t0=t0, duration=duration)
    tm = TPFModel(tpf)
    tm.include_model(transitmodel)
    tm.fit()
#    assert np.isclose(tm.bkg0, 100, rtol=3).all()
    assert tm.minframe.shape == tpf.shape[1:]
    tm.make_pdf()
