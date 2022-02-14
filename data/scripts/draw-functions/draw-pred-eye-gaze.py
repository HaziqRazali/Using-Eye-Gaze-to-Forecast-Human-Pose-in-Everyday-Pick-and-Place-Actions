import random
import argparse
import numpy as np
import pandas as pd
import math
import json
import sys
import os

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt5.QtCore import QBuffer
from PyQt5 import QtCore, QtGui
from pyqtgraph.Qt import QtWidgets
from scipy.spatial.transform import Rotation as R

sys.path.append("..")
from utils_data import *
from utils_draw import *
from utils_processing import *

pg.setConfigOption('background', 'white')
              
if __name__ == '__main__':
    
    # python draw-pred-eye-gaze.py --frame 2600 --draw_true_pose 0 --draw_pred_pose 0 --object_attention 0 --gaze_attention 0
    # python draw-pred-eye-gaze.py --frame 2600 --draw_true_pose 0 --draw_pred_pose 1 --object_attention 1 --gaze_attention 1
    # python draw-pred-eye-gaze.py --frame 2180 --draw_true_pose 0 --draw_pred_pose 0 --draw_grid 0 --gaze_attention 0
    # python draw-pred-eye-gaze.py --frame 2180 --draw_true_pose 0 --draw_pred_pose 1 --draw_grid 1 --gaze_attention 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="../../..", type=str)
    parser.add_argument('--result_root', default="results/ICRA2022", type=str)
    parser.add_argument('--result_name', default="vanilla", type=str)
    parser.add_argument('--sequence', default="p1_1", type=str)
    parser.add_argument('--frame', default=2600, type=int)
    
    parser.add_argument("--draw_inp_pose", choices=[0,1], default=1, type=int)
    parser.add_argument("--draw_pred_pose", choices=[0,1], default=1, type=int)
    parser.add_argument("--draw_true_pose", choices=[0,1], default=0, type=int)
    
    parser.add_argument("--draw_objects", choices=[0,1], default=1, type=int)
    
    parser.add_argument("--default_object_color", choices=[0,1], default=1, type=int)
    parser.add_argument("--object_attention", choices=[0,1], default=1, type=int)
    parser.add_argument("--color_pred_key_object", choices=[0,1], default=0, type=int, help="Gives a hardcoded color to the pred key object. Overwrites object_color and object_attention")
    parser.add_argument("--color_true_key_object", choices=[0,1], default=0, type=int, help="Gives a hardcoded color to the true key object. Overwrites color_pred_key_object, object_color, and object_attention")
    parser.add_argument("--draw_gaze", choices=[0,1], default=1, type=int)
    parser.add_argument("--gaze_attention", choices=[0,1], default=1, type=int)
    parser.add_argument("--draw_grid", choices=[0,1], default=1, type=int)
    
    parser.add_argument("--save_figure", choices=[0,1], default=0, type=int)
    
    args = parser.parse_args()
    
    # initialize screen
    app = QtWidgets.QApplication(sys.argv)
    w = gl.GLViewWidget()
            
    sequence = args.sequence
    filename = args.frame # 0000002555 0000002600 4447 4376
    filename = str(filename).zfill(10)
    results_dir = os.path.join(args.root,args.result_root,args.result_name,sequence)    
    pose_dir    = os.path.join(args.root,"data","xyz-poses",sequence)
    object_dir  = os.path.join(args.root,"data","object-positions-orientations",sequence)
    
    print(os.path.join(pose_dir,filename))
    print(os.path.join(object_dir,filename))
    print()
        
    # # # # # # #
    # Load data #
    # # # # # # #

    result = json.load(open(os.path.join(results_dir,filename+".json"),"r"))
    print(result["instruction"])
    print("Action=",result["action"], "\n")
    
    # # # # # # #
    # Draw Pose #
    # # # # # # #
    
    if args.draw_inp_pose:
        w = draw_pose(result["inp_pose"], result["center"], w, (1,1,1,1), link_ids)
    if args.draw_pred_pose:
        w = draw_pose(result["pred_key_pose"], result["center"], w, (0,0,1,1), link_ids)
    if args.draw_true_pose:
        w = draw_pose(result["true_key_pose"], result["center"], w, (1,0,0,1), link_ids)
    
    if result["action"] == "place":
        
        all_objects = pd.read_csv(os.path.join(object_dir,str(result["inp_frame"]).zfill(10)+".txt"),sep=" ")
                
        # get the grid coordinates and their scores
        
        pred_object_scores = np.array(result["pred_object_scores"])
        object_coordinates = np.array(result["objects"]) + np.array(result["center"])
                        
        # # # # # # # # # #
        # Draw Furnitures #
        # # # # # # # # # #
        
        furnitures = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table") | all_objects['name'].str.contains("chair")]
        for idx,furniture in furnitures.iterrows(): 
                
            furniture_cloud_name = furniture["name"]
          
            # load furniture meshfile
            with open(os.path.join(args.root,"data","meshes","json",furniture_cloud_name+".json"), 'r') as fp:
                data = json.load(fp)
            faces = np.array(data["faces"])
            vertices = np.array(data["vertices"])[:,:3]

            # furniture color
            colors = np.repeat(np.array([[0.7, 0.7, 0.7, 0.85]]), faces.shape[0], axis=0)
                                
            # transform furniture
            translation = furniture[["x","y","z"]].values        
            rotation = furniture[["a","b","c","d"]].values
            vertices = transform_object(vertices, translation=translation, rotation=rotation)
            
            # Mesh item will automatically compute face normals.
            object_renderer = gl.GLMeshItem(vertexes=vertices, faces=faces, faceColors=colors, smooth=False)
            object_renderer.setGLOptions('translucent')
            w.addItem(object_renderer)
    
        # # # # # # # # #
        # Draw Objects  #
        # # # # # # # # #
                    
        if args.draw_objects:
        
            objects = all_objects[~all_objects['name'].str.contains("shelf") & ~all_objects['name'].str.contains("table") & ~all_objects['name'].str.contains("chair")]
            for object_df_idx in objects.index.tolist():
                
                    object = objects.loc[object_df_idx]
                    object_cloud_name = object["name"]
                    object_cloud_color = get_colour(object_cloud_name, OBJECT_COLOURS)
                  
                    # load object meshfile
                    with open(os.path.join(args.root,"data","meshes","json",object_cloud_name+".json"),"r") as fp:
                        data = json.load(fp)
                    faces = np.array(data["faces"])
                    vertices = np.array(data["vertices"])[:,:3]

                    # initialize object color
                    colors = np.array([[object_cloud_color[0],object_cloud_color[1],object_cloud_color[2],1]]) if args.default_object_color else np.array([[0,1,0,1]])
                    colors = np.repeat(colors,faces.shape[0], axis=0)
                    #colors[:,3] = colors[:,3]*pred_object_score*5 if args.object_attention and f >= 5 else colors[:,3]
                    
                    # transform object
                    translation = object[["x","y","z"]].values        
                    rotation = object[["a","b","c","d"]].values
                    vertices = transform_object(vertices, translation=translation, rotation=rotation)
                    
                    # Mesh item will automatically compute face normals.
                    object_renderer = gl.GLMeshItem(vertexes=vertices, faces=faces, faceColors=colors, smooth=False)
                    object_renderer.setGLOptions('translucent')
                    w.addItem(object_renderer) 
    
        # # # # # # #
        # Draw Grid #
        # # # # # # #
        
        if args.draw_grid:
        
            # scale the colors of top N by a certain amount
            # scale the colors of after top N by another certain amount
            pred_object_scores_sorted_ind = pred_object_scores.argsort()
            pred_object_scores = pred_object_scores[pred_object_scores_sorted_ind[::-1]]
            object_coordinates = object_coordinates[pred_object_scores_sorted_ind[::-1]]
            
            scaled_pred_object_scores = np.copy(pred_object_scores)
            start_end_scale_list = [[0,6,10],[6,15,2],[15,200,1]] # [start,end,scale]
            for start,end,scale in start_end_scale_list:
                scaled_pred_object_scores[start:end] = scaled_pred_object_scores[start:end]*scale
                    
            colors = [[0,1,0]]*object_coordinates.shape[0]
            colors = np.stack(colors)
            colors = np.concatenate((colors,np.expand_dims(scaled_pred_object_scores,axis=1)),axis=1)
            size = np.ones(object_coordinates.shape[0])*15
                
            grid_renderer = gl.GLScatterPlotItem(pos=object_coordinates, color=colors, size=size)
            w.addItem(grid_renderer)
    
        # # # # # # # # #
        # Draw Eye Gaze #
        # # # # # # # # #

        gaze_scores = np.array(result["gaze_scores"]) # [num objects (10), gaze length, 1]
        gaze_scores = gaze_scores[np.argmax(pred_object_scores)]
        print("\ngaze_scores")
        for gaze_score in gaze_scores:
            print(gaze_score*5) 
        for i in range(len(result["gaze_vector"])):
            w = draw_gaze_np(result["gaze_vector"][i], result["center"], w, (1,0,1,1), width=2) if not args.gaze_attention else draw_gaze_np(result["gaze_vector"][i], result["center"], w, (1,0,1,gaze_scores[i]*5), width=2)
    
    if result["action"] == "pick":
                    
        all_objects = pd.read_csv(os.path.join(object_dir,str(result["key_frame"]).zfill(10)+".txt"),sep=" ")
    
        # # # # # # # # # # # # # # # # # # #  
        # get the true and pred key objects #
        # # # # # # # # # # # # # # # # # # #
        
        pred_object_scores = result["pred_object_scores"]   # predicted object scores
        pred_key_object_idx = np.argmax(pred_object_scores) # predicted key object id from the neural network ### TODO: CHECK ITS OUTPUT IF ITS A PLACE ACTION ###
        object_df_idxs = np.array(result["object_indexes"]).astype(np.int32) # object ids from the dataframe
        pred_key_object_df_idx = object_df_idxs[pred_key_object_idx] # predicted key object id from the dataframe
        true_key_object_idx = result["true_object_scores"] # true key object id from the dataframe
        print("pred_object_scores")
        for pred_object_score in pred_object_scores:
            print(pred_object_score)
        print()
        
        print("pred_key_object_idx", pred_key_object_idx)
        print("pred_key_object_df_idx", pred_key_object_df_idx)
        
        # # # # # # # # # #
        # Draw Furnitures #
        # # # # # # # # # #
        
        # this section just renders the furnitures
        furnitures = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table") | all_objects['name'].str.contains("chair")]
        for idx,furniture in furnitures.iterrows(): 
                
            furniture_cloud_name = furniture["name"]
          
            # load furniture meshfile
            with open(os.path.join(args.root,"data","meshes","json",furniture_cloud_name+".json"), 'r') as fp:
                data = json.load(fp)
            faces = np.array(data["faces"])
            vertices = np.array(data["vertices"])[:,:3]

            # furniture color
            colors = np.repeat(np.array([[0.7, 0.7, 0.7, 0.85]]), faces.shape[0], axis=0)
                                
            # transform furniture
            translation = furniture[["x","y","z"]].values        
            rotation = furniture[["a","b","c","d"]].values
            vertices = transform_object(vertices, translation=translation, rotation=rotation)
            
            # Mesh item will automatically compute face normals.
            object_renderer = gl.GLMeshItem(vertexes=vertices, faces=faces, faceColors=colors, smooth=False)
            object_renderer.setGLOptions('translucent')
            w.addItem(object_renderer)
        
        # # # # # # # # # # # # # # # #
        # Draw Object and Key-Objects #
        # # # # # # # # # # # # # # # #
                
        # load key object and objects 
        objects = all_objects[~all_objects['name'].str.contains("shelf") & ~all_objects['name'].str.contains("table") & ~all_objects['name'].str.contains("chair")]
        true_key_object_idx = result["true_object_scores"]
        print(objects)
        
        for object_df_idx,pred_object_score in zip(object_df_idxs,pred_object_scores):
            
            object = objects.loc[object_df_idx]
            object_cloud_name = object["name"]
            object_cloud_color = get_colour(object_cloud_name, OBJECT_COLOURS)
            print("object", object_cloud_name, "idx", object_df_idx, "object score", pred_object_score)
          
            # load object meshfile
            with open(os.path.join(args.root,"data","meshes","json",object_cloud_name+".json"),"r") as fp:
                data = json.load(fp)
            faces = np.array(data["faces"])
            vertices = np.array(data["vertices"])[:,:3]

            # initialize object color
            colors = np.array([[object_cloud_color[0],object_cloud_color[1],object_cloud_color[2],1]]) if args.default_object_color else np.array([[0,1,0,1]])
            colors = np.repeat(colors,faces.shape[0], axis=0)
            colors[:,3] = colors[:,3]*pred_object_score*5 if args.object_attention else colors[:,3]

            # color_pred_key_object overwrite initial object color
            if object_df_idx == pred_key_object_df_idx and args.color_pred_key_object:
                colors = np.array([[1,1,1,1]])
                colors = np.repeat(colors,faces.shape[0], axis=0)
                #colors[:,:3] = colors[:,:3]*pred_object_score*2 if args.object_attention else colors[:,:3]

            # color_true_key_object overwrite all 
            if object_df_idx == true_key_object_idx and args.color_true_key_object:
                colors = np.array([[1,0,0,1]])
                colors = np.repeat(colors,faces.shape[0], axis=0)
            
            # transform object
            translation = object[["x","y","z"]].values        
            rotation = object[["a","b","c","d"]].values
            vertices = transform_object(vertices, translation=translation, rotation=rotation)
            
            # Mesh item will automatically compute face normals.
            object_renderer = gl.GLMeshItem(vertexes=vertices, faces=faces, faceColors=colors, smooth=False)
            object_renderer.setGLOptions('translucent')
            w.addItem(object_renderer)
            
    # # # # # # # # #
    # Draw Eye Gaze #
    # # # # # # # # #
        
    if args.draw_gaze:
        gaze_scores = np.array(result["gaze_scores"]) # [num objects (10), gaze length, 1]
        gaze_scores = gaze_scores[np.argmax(pred_object_scores)]
        print("\ngaze_scores")
        for gaze_score in gaze_scores:
            print(gaze_score*5) 
        for i in range(len(result["gaze_vector"])):
            w = draw_gaze_np(result["gaze_vector"][i], result["center"], w, (1,0,1,1), width=2) if not args.gaze_attention else draw_gaze_np(result["gaze_vector"][i], result["center"], w, (1,0,1,gaze_scores[i]*5), width=2) 
        
        fixation_file = os.path.join(args.root,"data","fixations",sequence+".csv")
        fixation_all = pd.read_csv(fixation_file)
        print(fixation_all.iloc[args.frame])
    
    from pyqtgraph.Qt import QtCore, QtGui
    
    if args.save_figure:
        d = w.renderToArray((1000, 1000))        
        lcrop = 400
        rcrop = 300
        tcrop = 300
        bcrop = 350
        d = d[lcrop:-rcrop,tcrop:-bcrop]
        print("Saving", str(args.frame)+".png")
        pg.makeQImage(d).save(str(args.frame)+".png")
    
    else:
        #original settings
        g = gl.GLGridItem(size=QtGui.QVector3D(100,100,1),color=(255, 255, 0, 100))
        w.addItem(g)    
        
        axes = gl.GLAxisItem()
        axes.setSize(2,2,2)
        #w.addItem(axes)
        w.show()
        w.orbit(0,0)
        #w.pan(-3,-3,-3) #blue, yellow, green
        app.exec()
        
        