from .srn import SRNDataset
from .srn_priors import SRNPriorsDataset
from .co3d import CO3DDataset
from .nmr import NMRDataset
from .objaverse import ObjaverseDataset
from .gso import GSODataset

SHAPENET_DATASET_ROOT = "/content/SRN" # Change this to your data directory
CO3D_DATASET_ROOT = None # Change this to where you saved preprocessed data
NMR_DATASET_ROOT = None # Change this to your data directory
OBJAVERSE_ROOT = None # Change this to your data directory
OBJAVERSE_LVIS_ANNOTATION_PATH = None # Change this to your filtering .json path
GSO_ROOT = None # Change this to your data directory

def get_dataset(cfg, name):
    if cfg.data.category == "cars" or cfg.data.category == "chairs":
        assert SHAPENET_DATASET_ROOT is not None, "Update the location of the SRN Shapenet Dataset"        
        return SRNDataset(cfg, SHAPENET_DATASET_ROOT, name)
    if cfg.data.category == "cars_priors":
        return SRNPriorsDataset(cfg, name)
    elif cfg.data.category == "hydrants" or cfg.data.category == "teddybears":        
        assert CO3D_DATASET_ROOT is not None, "Update the location of the CO3D Dataset"
        return CO3DDataset(cfg, CO3D_DATASET_ROOT, name)
    elif cfg.data.category == "nmr":        
        assert NMR_DATASET_ROOT is not None, "Update path of the dataset"
        return NMRDataset(cfg, name)
    elif cfg.data.category == "objaverse":        
        assert OBJAVERSE_ROOT is not None, "Update dataset path"
        assert OBJAVERSE_LVIS_ANNOTATION_PATH is not None, "Update filtering .json path"
        return ObjaverseDataset(cfg, name)
    elif cfg.data.category == "gso":
        assert GSO_ROOT is not None, "Update path of the dataset"
        return GSODataset(cfg, name)