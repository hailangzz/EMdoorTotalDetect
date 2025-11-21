import numpy as np

anchors = np.load("anchors_192.npy")  # shape (2016, 4)
anchors.astype(np.float32).tofile("anchors_192.bin")