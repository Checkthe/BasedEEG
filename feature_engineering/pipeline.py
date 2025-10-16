from calculate_feature import start_cf
from generate_and_combine import start_gac
from chose_and_scale import start_cas
from RandomForestClassifier import start_RFC

if __name__ == "__main__":
    from config import cfg
    from utils.loading_feature import load_2b, load_2a
    from utils.loading import loaddata_singal, loaddata_BCICIV2B, loaddata_BCICIV2A
    cfgs = cfg().get_args()
    ds = load_2b()
    dataset = loaddata_BCICIV2B(crt=False)
    start_cf(dataset=dataset)
    start_gac(ds=ds)
    start_cas()
    start_RFC()