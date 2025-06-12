import os
import logging

# set up global logger
time_format = '%H:%M:%S'
log_format = '[%(levelname)s] %(asctime)s.%(msecs)03d [%(filename)s]: %(message)s'
logging.basicConfig(format=log_format, datefmt=time_format)

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger.info("Logging initialized")

def get_root_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_index = script_dir.find('/src')
    root_dir = script_dir[:src_index]
    return root_dir

def get_common_paths():
    root_dir = get_root_dir()
    data_path = os.path.join(root_dir, "data/star_classification.csv")
    hist_dir_path = os.path.join(root_dir, "results/histograms")
    default_dir_path = os.path.join(root_dir, "results/default")
    logreg_dir_path = os.path.join(root_dir, "results/logreg")
    nn_dir_path = os.path.join(root_dir, "results/nn")
    knn_dir_path = os.path.join(root_dir, "results/knn")
    
    # make sure destination dirs exist
    os.makedirs(hist_dir_path, exist_ok=True)
    os.makedirs(default_dir_path, exist_ok=True)
    os.makedirs(logreg_dir_path, exist_ok=True)
    os.makedirs(nn_dir_path, exist_ok=True)
    os.makedirs(knn_dir_path, exist_ok=True)
    
    return {'data_path': data_path,
            'hist_dir_path': hist_dir_path,
            'default_dir_path': default_dir_path,
            'logreg_dir_path': logreg_dir_path,
            'nn_dir_path': nn_dir_path,
            'knn_dir_path': knn_dir_path}
