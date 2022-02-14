import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# # # # # # # # # # # # # # # # # # # # # # #
# converts exp to xyz representation        #
# - won't be able to recover initial offset #
# # # # # # # # # # # # # # # # # # # # # # #

def expmap_to_xyz(exp_seq, parents, bone_lengths, reference_vector=[0,-1,0]):

    # print(exp_seq.shape) # [T, 21, 3]

    reference_vector = np.array(reference_vector)
    assert exp_seq.ndim >= 2 and exp_seq.shape[-1] == 3, \
        "Wanted TxJx3 array containing T expmap skeletons, each with J dirs."
            
    toposorted = toposort(parents)
    
    root = toposorted[0] # head    
    xyz_seq = np.zeros_like(exp_seq) # [T, 21, 3]
    
    # restore head first i.e. recover the original position by computing the cumulative sum
    xyz_seq[:, root, :] = np.cumsum(exp_seq[:, root, :], axis=0)
        
    # simultaneously recover bones (normalised offset from parent) and original
    # coordinates
    bones = np.zeros_like(exp_seq) # [T, 21, 3]
    
    # why fix the bone to be 1 ? 
    # - because we set it as the initial reference vector at xyz_to_exp()
    bones[:, root] = np.repeat(reference_vector[np.newaxis, :], bones.shape[0], axis=0) # [T, 21, 3]
    
    # process from the root to the child
    for child in toposorted[1:]:
        
        # get parent
        parent = parents[child]        
        #print("child=",joint_names[child],", parents[child]=",joint_names[parents[child]])
        
        # get position of parent's bone (reference_vector e.g. [0, -1, 0] if root)
        parent_bone = bones[:, parent, :]   # [T, 3]
                
        # get exponential coordinates of the current joint being processed
        exps = exp_seq[:, child, :]         # [T, 3]        
        
        # loop through each timestep
        for t in range(len(exp_seq)):
            
            # compute the rotation matrix given the exponential map
            R = exp_to_rotmat(exps[t]) # [3, 3]
                        
            # rotate the bone
            bones[t, child] = np.dot(R, parent_bone[t]) # [3]
            #print(np.dot(R, parent_bone[t])) == print(np.matmul(R,parent_bone[t]))
            
        # scale the bone
        # !!!!!!!!!!!!!!!!!!!!!!!!!! we must now go in the opposite direction because we previously computed the vector from the child to the parent
        scaled_child_bones = -bones[:, child] * bone_lengths[child]
        
        # coordinates of parent + scaled_child_bones vector to obtain bone position of child
        xyz_seq[:, child] = xyz_seq[:, parent] + scaled_child_bones

    return xyz_seq
    
def exp_to_rotmat(exp):
    """Convert rotation paramterised as exponential map into ordinary 3x3
    rotation matrix."""
    assert exp.shape == (3, ), "was expecting expmap vector"

    # begin by normalising all exps
    # we previously encoded the exponential information as follows
    # - [rotation_x, rotation_y, rotation_z] * angle
    # - the angle is thus reobtained by computing the L2 norm
    angle = np.linalg.norm(exp)
    if angle < 1e-5:
        # assume no rotation
        return np.eye(3)
    
    # the direction is then the exponential information divided by the angle
    dir = exp / angle

    # Rodrigues' formula, matrix edition
    K = np.array([[0, -dir[2], dir[1]], [dir[2], 0, -dir[0]],
                  [-dir[1], dir[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

# # # # # # # # # # # # # # # # # # # # 
# converts xyz to exp representation  #
# # # # # # # # # # # # # # # # # # # #
def xyz_to_exp(xyz_seq, parents, reference_vector=[0,-1,0]):
    
    # print(xyz_seq.shape) # (N, 21, 3) 
    reference_vector = np.array(reference_vector)
    
    """Converts a tree of (x, y, z) positions into the parameterisation used in
    the SRNN paper, "modelling human motion with binary latent variables"
    paper, etc. Stores inter-frame offset in root joint position."""
    assert xyz_seq.ndim == 3 and xyz_seq.shape[2] == 3, \
        "Wanted TxJx3 array containing T skeletons, each with J (x, y, z)s"

    exp_seq = np.zeros_like(xyz_seq)
    
    # Return toposorted array of joint indices (sorted root-first)
    toposorted = toposort(parents)
    
    # [1:] ignores the root; apart from that, processing order doesn't actually matter
    for child in toposorted[1:]:
        parent = parents[child]
                
        #print("child=",joint_names[child],", parents[child]=",joint_names[parents[child]])
        
        # bones is the vector that points from child to parent
        bones = xyz_seq[:, parent] - xyz_seq[:, child] # [N, 3]
        grandparent = parents[parent]
        #print("grandparent=", joint_names[grandparent])
        
        if grandparent == parent:
            # we're the root; parent bones will be constant (x,y,z)=(0,1,0)
            parent_bones = np.repeat(reference_vector[np.newaxis, :], bones.shape[0], axis=0) # [N, 3]    
            # todo - why (x,y,z) = [0,1,0]
            #      - is it something to do with the angle ? (CORRECT)
            #      - is it to keep a reference angle ? (CORRECT)
            #      - simply used as a reference, later when going from expmap to xyz we set the parent bone to be the same i.e. (0,1,0)
        else:
            # we actually have a parent bone :)
            parent_bones = xyz_seq[:, grandparent] - xyz_seq[:, parent] # [N, 3]
            
        # normalise parent and child bones
        norm_bones = _norm_bvecs(bones)
        norm_parent_bones = _norm_bvecs(parent_bones)
        
        # cross product will only be used to get axis around which to rotate
        # - recall that cross product returns a vector that is perpendicular to both input vectors
        # - why do we get the cross product between [child->parent, parent->grandparent] bone ?
        #   - does not matter. We simply want the angle between the two. 
        cross_vecs = np.cross(norm_parent_bones, norm_bones)    # [N, 3]
        norm_cross_vecs = _norm_bvecs(cross_vecs)               # [N, 3]
        
        # arccos of the dot product to get the rotation angle
        # - recall that the dot product gives us the angle between the 2 input vectors
        # - thus, the parent-grandparent bone direction can be obtained by rotating the child-parent bone by angles about norm_cross_vecs, a 3 dimensional vector
        dot = np.sum(norm_bones * norm_parent_bones, axis=-1)   # [N]
        dot[dot > 1.0] = 1.0
        angles = np.arccos(dot)                                 # [N]
        assert not np.isnan(angles).any()
        
        # relate this to the fact that the norm of each row of log_map = angle, later used in exp_to_rotmat
        # - i think it is simply how the authors decide to store the exp information
        #   - [vector_x, vector_y, rotation_z] * angle 
        #   - i.e. the length of the vector encodes the angle
        #   - and we obtain the angle through the L2 norm
        log_map = norm_cross_vecs * angles[..., None] # [T, 3]     
        #print(norm_cross_vecs[0], angles[0])
        #print(np.linalg.norm(log_map,axis=-1)[0]) # = angles[0]
        
        exp_seq[:, child] = log_map
        
    # root will store distance from previous frame
    root = toposorted[0]
    exp_seq[1:, root] = xyz_seq[1:, root] - xyz_seq[:-1, root]
    
    return exp_seq

def _norm_bvecs(bvecs):
    """Norm bone vectors, handling small magnitudes by zeroing bones."""
    bnorms = np.linalg.norm(bvecs, axis=-1)
    mask_out = bnorms <= 1e-5
    # implicit broadcasting is deprecated (?), so I'm doing this instead
    _, broad_mask = np.broadcast_arrays(bvecs, mask_out[..., None])
    bvecs[broad_mask] = 0
    bnorms[mask_out] = 1
    return bvecs / bnorms[..., None]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
# Return toposorted array of joint indices (sorted root-first)  # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def toposort(parents):
    toposorted = []
    visited = np.zeros_like(parents, dtype=bool)
    for joint in range(len(parents)):
        if not visited[joint]:
            _toposort_visit(parents, visited, toposorted, joint)

    check_toposorted(parents, toposorted)

    return np.asarray(toposorted)

def _toposort_visit(parents, visited, toposorted, joint):
    parent = parents[joint]
    visited[joint] = True
    if parent != joint and not visited[parent]:
        _toposort_visit(parents, visited, toposorted, parent)
    toposorted.append(joint)

def check_toposorted(parents, toposorted):

    # check that array contains all/only joint indices
    assert sorted(toposorted) == list(range(len(parents)))

    # make sure that order is correct
    to_topo_order = {
        joint: topo_order
        for topo_order, joint in enumerate(toposorted)
    }
    for joint in toposorted:
        assert to_topo_order[joint] >= to_topo_order[parents[joint]]

    # verify that we have only one root
    joints = range(len(parents))
    assert sum(parents[joint] == joint for joint in joints) == 1
    
# # # # # # # # # # # # #
# compute bone lengths  # 
# # # # # # # # # # # # #
def bone_lengths(poses, parents):
    lengths = np.zeros(poses.shape[:-1]) # [N, 21]
    
    for child in toposort(parents)[1:]:
        parent = parents[child]
        child_locs = poses[..., child, :]
        parent_locs = poses[..., parent, :]
        diffs = parent_locs - child_locs            # [N, 3]
        diffs = np.sqrt(np.sum(diffs**2,axis=-1))   # [N]
        cp_lengths = diffs
        lengths[..., child] = cp_lengths
    
    return lengths
    
# # # # # # # # # # # # #
# loads the human pose  #
# # # # # # # # # # # # #
def load_pose(filename):
    pose = pd.read_csv(filename, sep=" ")
    pose = pose[["worldLinkFramePosition_x","worldLinkFramePosition_y","worldLinkFramePosition_z"]].values.astype(np.float32)
    return pose
    
# # # # # # # # # # # # # # # # # # # #
# get the object noun given its name  #
# # # # # # # # # # # # # # # # # # # # 
def get_noun(name):
    for x in OBJECT_NAMES:
        if x in name:
            return x
            
def get_colour(name, colours):
    for k,v in colours.items():
        if k in name:
            return v
    
    if "bowl" in name:
        return colours["white"]
        
    if "jug" in name:
        return colours["dark_green"]
   
# loads the object cloud and color
####################################
def load_cloud(filename,size=0.1,color=[1.0,0.0,0.0,0.5]):
    cloud_pos   = pd.read_csv(filename).values
    cloud_size  = np.ones(shape=cloud_pos.shape[0])*size
    cloud_color = np.ones(shape=(cloud_pos.shape[0],4))*color
    return [cloud_pos, cloud_size, cloud_color]

# # # # # # # # # # # # #
# loads the gaze vector #
# # # # # # # # # # # # #
def load_gaze(filename):
    gaze = pd.read_csv(filename, sep=",")
    gaze_p1 = gaze[["x1","y1","z1"]].to_numpy().squeeze()
    gaze_p2 = gaze[["x2","y2","z2"]].to_numpy().squeeze()
    return np.stack((gaze_p1,gaze_p2))
    
# # # # # # # # # # # # #
# loads the human pose  #
# # # # # # # # # # # # #
def load_human(filename,sep):
    
    # read dataframe
    df_old = pd.read_csv(filename,sep=sep)
    
    # create new dataframe
    df_new = pd.DataFrame(columns=df_old.columns)
    
    df_joints = [None]*len(joints)
    for i,joint in enumerate(joints):
        df_joints[i] = df_old[df_old["name"].str.startswith(joint)]
        df_joints[i] = df_joints[i].median(axis=0).to_frame().T
        df_joints[i]["name"] = joint
    
    df_new = pd.concat(df_joints)
    return df_new
 
# rotates then translates the object wrt to world coordinates
####################################    
def transform_object(vertices, translation, rotation):
       
    translation=np.squeeze(translation)
    rotation=np.squeeze(rotation)
          
    # get rotation matrix
    rotation = R.from_quat(rotation).as_matrix()
        
    # center
    #center = np.mean(vertices,axis=0)
        
    # rotate about origin first then translate
    #vertices = np.matmul(rotation,(vertices-center).T).T + center + translation
    vertices = np.matmul(rotation,(vertices).T).T + translation
    return vertices