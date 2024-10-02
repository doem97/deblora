# Define the global label mapping
LABEL_MAPPING = {
    "ship": (0, "head"),
    "small-vehicle": (1, "head"),
    "large-vehicle": (2, "head"),
    "plane": (3, "head"),
    "harbor": (4, "middle"),
    "storage-tank": (5, "middle"),
    "tennis-court": (6, "middle"),
    "bridge": (7, "middle"),
    "swimming-pool": (8, "middle"),
    "helicopter": (9, "tail"),
    "basketball-court": (10, "tail"),
    "baseball-diamond": (11, "tail"),
    "roundabout": (12, "tail"),
    "soccer-ball-field": (13, "tail"),
    "ground-track-field": (14, "tail"),
}

# Define class groups
CLASS_GROUPS = {
    "head": [
        name for name, (_, group) in LABEL_MAPPING.items() if group == "head"
    ],
    "middle": [
        name for name, (_, group) in LABEL_MAPPING.items() if group == "middle"
    ],
    "tail": [
        name for name, (_, group) in LABEL_MAPPING.items() if group == "tail"
    ],
}
