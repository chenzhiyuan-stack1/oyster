import numpy as np

ref_poses = np.loadtxt('pace.txt')
dim1 = ref_poses.shape[0]
dim2 = ref_poses.shape[1]
ref_poses_1 = np.zeros([dim1, dim2, 1], dtype=float)
for i in range(dim1):
    cur_ref_pose = ref_poses[i]
    tmp = np.zeros([dim2, 1], dtype=float)
    for j in range(dim2):
        tmp[j] = np.array(cur_ref_pose[j])
    print(tmp)
    ref_poses_1[i] = tmp
print(ref_poses_1)