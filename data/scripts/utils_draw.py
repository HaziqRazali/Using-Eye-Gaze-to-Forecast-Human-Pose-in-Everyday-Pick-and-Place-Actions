import numpy as np
import pandas as pd
import pyqtgraph.opengl as gl
  
# # # # # # # # # # # #
# draw the human pose #
# # # # # # # # # # # #
def draw_pose(pose, center, w, color, link_ids):
    pose = np.array(pose)
    center = np.array(center)
    print(pose.shape)
    pose += center
    pose[:,-1] -= np.min(pose[:,-1]) # make sure foot is touching the ground
    
    for a,b in link_ids:
        x = pose[a] 
        y = pose[b]
        line = np.array([x,y])
        line_renderer = gl.GLLinePlotItem(pos=line, width=5, antialias=False, color=color)
        line_renderer.setGLOptions('opaque')
        w.addItem(line_renderer)
        
    return w
    
# draw the eye gaze
####################################
def draw_gaze(gaze, center, w, color):

    x = gaze[["x1","y1","z1"]].values.squeeze() + center
    y = gaze[["x2","y2","z2"]].values.squeeze() + center
    line1 = np.array([x,y])
    line_renderer = gl.GLLinePlotItem(pos=line1, width=5, antialias=False, color=color)
    line_renderer.setGLOptions('opaque')
    w.addItem(line_renderer)
    
    return w
    
# draw the eye gaze
####################################
def draw_gaze_np(gaze, center, w, color, width):
    
    gaze = gaze + np.array(center)
    
    gaze = np.array(gaze)
    line1 = np.array([gaze[0],gaze[1]])
    line_renderer = gl.GLLinePlotItem(pos=line1, width=width, antialias=False, color=color)
    #line_renderer.setGLOptions('opaque')
    w.addItem(line_renderer)
    
    return w