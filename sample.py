import numpy as np

file = r"CMU/CMU/01/01_01_poses.npz"   # pick any file you want
data = np.load(file, allow_pickle=True)

# Pick frame index
i = 0   # first frame (change this to any frame number)
frame_pose = data["poses"][i]      # shape: (156,) or (72,) depending on model
frame_trans = data["trans"][i]     # shape: (3,)
betas = data["betas"]              # shape: (16,) usually
#xyz rotations for each joint
print("Frame pose:", frame_pose)
#translation of the root joint
print("Frame trans:", frame_trans)
#body shape parameters to describe the body shape and reconstruct the mesh
print("Betas:", betas)
