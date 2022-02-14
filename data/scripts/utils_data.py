import numpy as np

OBJECT_NAMES = ["table","cup","shelf","plate","jug","plate","chair","bowl"]
OBJECT_COLOURS = {"red":np.array([1,0,0,1]), "green":np.array([0,1,0,1]), "dark_green":np.array([0,0.75,0,1]), "blue":np.array([0,0,1,1]), "pink":np.array([1,0.752,0.796,1]), "white":np.array([1,1,1,1])}

# # # # # # # # # # # #
# joint names and ids #
# # # # # # # # # # # #

joint_names = ["head","neck","torso","linnerShoulder","lShoulder","lElbow","lWrist","rinnerShoulder","rShoulder","rElbow","rWrist","pelvis","base","lHip","lKnee","lAnkle","lToe","rHip","rKnee","rAnkle","rToe"]
joint_ids = {name: idx for idx, name in enumerate(joint_names)}

# # # # # # # # # # # #
# link names and ids  #
# # # # # # # # # # # #

link_names = [["head","neck"],["neck","torso"],
["torso","linnerShoulder"],["linnerShoulder","lShoulder"],["lShoulder","lElbow"],["lElbow","lWrist"],
["torso","rinnerShoulder"],["rinnerShoulder","rShoulder"],["rShoulder","rElbow"],["rElbow","rWrist"],
["torso","pelvis"],["pelvis","base"],
["base","lHip"],["lHip","lKnee"],["lKnee","lAnkle"],["lAnkle","lToe"],
["base","rHip"],["rHip","rKnee"],["rKnee","rAnkle"],["rAnkle","rToe"]]

link_ids = [[0,1],[1,2],
[2,3],[3,4],[4,5],[5,6],
[2,7],[7,8],[8,9],[9,10],
[2,11],[11,12],
[12,13],[13,14],[14,15],[15,16],
[12,17],[17,18],[18,19],[19,20]]

# # # # # # # # # # # # # # # #
# joint parent names and ids  #
# # # # # # # # # # # # # # # #
joint_parent_names = {}
joint_parent_ids = {}
joint_parent_names["root"] = {}
joint_parent_ids["root"] = {}

# torso as root
joint_parent_names["root"]["torso"] = {# root
                                       "torso":"torso",
                                       # spinal column
                                       "neck":"torso","head":"neck","pelvis":"torso","base":"pelvis",
                                       # left half
                                       "linnerShoulder":"torso","lShoulder":"linnerShoulder","lElbow":"lShoulder","lWrist":"lElbow","lHip":"base","lKnee":"lHip","lAnkle":"lKnee","lToe":"lAnkle",
                                       # right half
                                       "rinnerShoulder":"torso","rShoulder":"rinnerShoulder","rElbow":"rShoulder","rWrist":"rElbow","rHip":"base","rKnee":"rHip","rAnkle":"rKnee","rToe":"rAnkle"}    
joint_parent_ids["root"]["torso"] = [joint_ids[joint_parent_names["root"]["torso"][child_name]] for child_name in joint_names]

# pelvis as root
joint_parent_names["root"]["pelvis"] = {# root
                                        "pelvis":"pelvis",                                        
                                        # spinal column
                                        "neck":"torso","head":"neck","torso":"pelvis","base":"pelvis",
                                        # left half
                                        "linnerShoulder":"torso","lShoulder":"linnerShoulder","lElbow":"lShoulder","lWrist":"lElbow","lHip":"base","lKnee":"lHip","lAnkle":"lKnee","lToe":"lAnkle",
                                        # right half
                                        "rinnerShoulder":"torso","rShoulder":"rinnerShoulder","rElbow":"rShoulder","rWrist":"rElbow","rHip":"base","rKnee":"rHip","rAnkle":"rKnee","rToe":"rAnkle"}
joint_parent_ids["root"]["pelvis"] = [joint_ids[joint_parent_names["root"]["pelvis"][child_name]] for child_name in joint_names]

# head as root
joint_parent_names["root"]["head"] = {# root
                                      "head":"head",                                        
                                      # spinal column
                                      "neck":"head","torso":"neck","pelvis":"torso","base":"pelvis",                                        
                                      # left half
                                      "linnerShoulder":"torso","lShoulder":"linnerShoulder","lElbow":"lShoulder","lWrist":"lElbow","lHip":"base","lKnee":"lHip","lAnkle":"lKnee","lToe":"lAnkle",                                        
                                      # right half
                                      "rinnerShoulder":"torso","rShoulder":"rinnerShoulder","rElbow":"rShoulder","rWrist":"rElbow","rHip":"base","rKnee":"rHip","rAnkle":"rKnee","rToe":"rAnkle"}    
joint_parent_ids["root"]["head"] = [joint_ids[joint_parent_names["root"]["head"][child_name]] for child_name in joint_names]

# # # # # # # # # # # # 
# bone median lengths #
# # # # # # # # # # # #
# head root
"""bone_median_lengths = [0., 0.14114, 0.23359123, 0.18687686, 0.1528039, 0.24337935,
 0.25898139, 0.18672664, 0.1528039, 0.24337935, 0.26657301, 0.20057803,
 0.07405368, 0.09030937, 0.38335675, 0.35157481, 0.1476358, 0.09030938,
 0.38335675, 0.34291348, 0.14763582]"""

# pelvis root
bone_median_lengths = [0.14984854, 0.23465495, 0.20130946, 0.18687688, 0.14682457, 0.26710218, 
                       0.24382357, 0.18672666, 0.14682457, 0.26710218, 0.24513078, 0.,
                       0.07862381, 0.09588271, 0.38335675, 0.41587715, 0.15674697, 0.09588271,
                       0.38335675, 0.41943094, 0.15674697]