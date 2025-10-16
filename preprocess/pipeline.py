from preprocess.filter import filtering
from preprocess.downsample import fir_resample
from preprocess.car import apply_car

def prep_pipeline(data,sf=250,fr=None,br=None):
    # data = fir_resample(data,orig_sfreq=sf,do_fir=True)
    data = filtering(data, sfreq=sf, bp_range=fr, bs_range=br)
    data = apply_car(data)
    return data