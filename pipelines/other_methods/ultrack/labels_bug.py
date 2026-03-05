from ultrack import MainConfig, Tracker
import numpy as np

labels = np.zeros((2, 50, 50), dtype=np.uint16)
labels[:, 10:15, 10:15] = 1
labels[:, 30:35, 30:35] = 2

# Create a config
config = MainConfig()

# this removes irrelevant segments from the image
# see the configuration section for more details
config.segmentation_config.min_frontier = 0.5
config.segmentation_config.min_area = 1

# Run the tracking
tracker = Tracker(config=config)
tracker.track(labels=labels)

# # Visualize the results
# tracks, graph = tracker.to_tracks_layer()
# napari.view_tracks(tracks[["track_id", "t", "y", "x"]], graph=graph)
# napari.run()