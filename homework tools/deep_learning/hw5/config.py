TRAIN_PATH = "data/train"
VALID_PATH = "data/valid"

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]
LABEL_NAMES_SHORT = ["bg", "ka", "pi", "ni", "bo", "pr"]

# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_TRAIN_PATH = "dense_data/train"
DENSE_VALID_PATH = "dense_data/valid"

DENSE_LABEL_NAMES = ["background", "kart", "track", "bomb/projectile", "pickup/nitro"]
DENSE_LABEL_NAMES_SHORT = ["bg", "ka", "tr", "bp", "pn"]
# DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]
DENSE_CLASS_DISTRIBUTION = [0.05, 0.035, 0.05, 0.006, 0.005]

DET_LABEL_NAMES = ["kart", "bomb/projectile", "pickup/nitro"]
DET_LABEL_NAMES_SHORT = ["ka", "bp", "pn"]
DET_CLASS_DISTRIBUTION = [0.035, 0.006, 0.005]

WANDB_PROJECT = {
    "cnn": "ai394-dl-hw3-cnn",
    "fcn": "ai394-dl-hw3-fcn",
    "det": "ai394-dl-hw4",
    "rl": "ai394-dl-hw5",
    "ec": "ai394-dl-hwec",
}
