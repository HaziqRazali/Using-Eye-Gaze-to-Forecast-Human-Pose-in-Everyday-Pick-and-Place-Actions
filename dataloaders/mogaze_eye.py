import os
import sys
import cv2
import time
import math
import h5py
import json
import torch
import pickle
import random
import numpy as np
import pandas as pd
import torchvision.transforms.functional as transforms

from ast import literal_eval
from tqdm import tqdm
from PIL import Image
from glob import glob
from random import randint
from scipy.spatial.transform import Rotation as R

def delete(data, substring):
    return [x for x in data if substring not in x]

def load_pose(filename):
    pose = pd.read_csv(filename, sep=" ")
    pose = pose[["worldLinkFramePosition_x","worldLinkFramePosition_y","worldLinkFramePosition_z"]].values.astype(np.float32)
    return pose

def load_gaze(filename):
    gaze = pd.read_csv(filename, sep=",")
    gaze_p1 = gaze[["x1","y1","z1"]].to_numpy().squeeze()
    gaze_p2 = gaze[["x2","y2","z2"]].to_numpy().squeeze()
    return np.stack((gaze_p1,gaze_p2))

train = ["p1_2","p2_1","p4_1","p5_1","p6_2","p7_3"]
val  = ["p1_1","p6_1","p7_1","p7_2"]
test = ["p1_1"]
synthetic = ["synthetic"]     
class dataloader(torch.utils.data.Dataset):
    def __init__(self, args, dtype):

        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.dtype = dtype

        # data
        pose_list,gaze_list,object_list,metadata_list,instruction_list,action_list,fixation_list = [],[],[],[],[],[],[]
        
        """
        Read human, object and segmentation data
        """
                                
        pose_folders       = delete(sorted(glob(os.path.join(self.dataset_root,"xyz-poses","*"))),"p3_1")                       # human pose
        gaze_folders       = delete(sorted(glob(os.path.join(self.dataset_root,"eye-gaze-processed","*"))),"p3_1")              # eye gaze
        object_folders     = delete(sorted(glob(os.path.join(self.dataset_root,"object-positions-orientations","*"))),"p3_1")   # position of every object
        metadata_folders   = delete(sorted(glob(os.path.join(self.dataset_root,"segmentations-processed","*"))),"p3_1")         # sequence name, inp frame, key frame, key-object name, 
        segment_files      = delete(sorted(glob(os.path.join(self.dataset_root,"segmentations","*"))),"p3_1")                   # segments of every sequence
        null_segments      = delete(sorted(glob(os.path.join(self.dataset_root,"null-segments","*"))),"p3_1")                   # null segments of every sequence
        erroneous_segments = delete(sorted(glob(os.path.join(self.dataset_root,"erroneous-segments","*"))),"p3_1")              # erroneous segments of every sequence
        instruction_files  = delete(sorted(glob(os.path.join(self.dataset_root,"instructions","*"))),"p3_1")                    # instructions for every sequence
        fixation_files     = sorted(glob(os.path.join(self.dataset_root,"fixations","*")))
             
        pose_folders       = [f for f in pose_folders for g in eval(dtype) if g in f]
        gaze_folders       = [f for f in gaze_folders for g in eval(dtype) if g in f]
        object_folders     = [f for f in object_folders for g in eval(dtype) if g in f]
        metadata_folders   = [f for f in metadata_folders for g in eval(dtype) if g in f]
        segment_files      = [f for f in segment_files for g in eval(dtype) if g in f]
        null_segments      = [f for f in null_segments for g in eval(dtype) if g in f]
        erroneous_segments = [f for f in erroneous_segments for g in eval(dtype) if g in f]
        instruction_files  = [f for f in instruction_files for g in eval(dtype) if g in f]
        fixation_files     = [f for f in fixation_files for g in eval(dtype) if g in f]
        
        print("Reading from", self.dataset_root, flush=True)
                
        max_pose_len = []
        for pose_folder,gaze_folder,object_folder,metadata_folder,segment_file,null_segment,erroneous_segment,instruction_file,fixation_file in zip(pose_folders,gaze_folders,object_folders,metadata_folders,segment_files,null_segments,erroneous_segments,instruction_files,fixation_files):
            
            """print(pose_folder)
            print(gaze_folder)
            print(object_folder)
            print(metadata_folder)
            print(segment_file)
            print(null_segment)
            print(instruction_file)
            print(fixation_file)"""
            print("Reading sequence " + pose_folder.split("/")[-1], flush=True)
            
            #if pose_folder.split("/")[-1] == "p1_2" or pose_folder.split("/")[-1] == "p2_1" or pose_folder.split("/")[-1] == "p4_1" or pose_folder.split("/")[-1] == "p5_1" or pose_folder.split("/")[-1] == "p6_2":
            #if pose_folder.split("/")[-1] == "p1_1" or pose_folder.split("/")[-1] == "p6_1" or pose_folder.split("/")[-1] == "p7_1":
            #    continue
                        
            # read in the list of text files
            pose_files_all     = sorted(glob(os.path.join(pose_folder,"*")))
            gaze_files_all     = sorted(glob(os.path.join(gaze_folder,"*")))
            object_files_all   = sorted(glob(os.path.join(object_folder,"*")))
            metadata_files_all = sorted(glob(os.path.join(metadata_folder,"*")))
            fixation_all = pd.read_csv(fixation_file)
                        
            assert len(pose_files_all) == len(gaze_files_all)
            assert len(pose_files_all) == len(object_files_all)
            assert len(object_files_all) == len(metadata_files_all)

            # # # # # # # # # # # # # # # # # # # # # # #
            # remove the start and end of each segment  #
            # # # # # # # # # # # # # # # # # # # # # # #
            
            non_null_entries = []
            segments = h5py.File(segment_file, 'r')['segments']
            for segment in segments:
                non_null_entries.extend(range(segment[0]+1,segment[1]-1))

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # remove first self.gaze_length * self.time_step_size frames so I don't index the gaze before the start of the sequence                                                          #
            # remove first inp_length * self.time_step_size and last self.out_length * self.time_step_size so I don't index the poses before the start or after the end of the sequence #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            after_first_M_frames = [i for i in range(self.gaze_length * self.time_step_size, len(pose_files_all))]
            after_first_N_frames = [i for i in range(self.inp_length * self.time_step_size, len(pose_files_all) - self.out_length * self.time_step_size)]

            # # # # # # # # # #
            # remove saccades #
            # # # # # # # # # #

            # collect frames that meet the fixation threshold
            valid_fixation_frames = fixation_all.index[fixation_all["fixation"] <= self.fixation_threshold].values

            # # # # # # # # # # # # # # # 
            # remove erroneous segments #
            # # # # # # # # # # # # # # #
            
            erroneous_frames = []
            f = open(erroneous_segment, "r")
            for x in f:
                x = x.split("-")
                l = int(x[0])
                r = int(x[1])
                erroneous_frames.extend(range(l,r+1)) 
            non_erroneous_frames = range(len(pose_files_all)+1)
            non_erroneous_frames = list(set(non_erroneous_frames) - set(erroneous_frames))

            # # # # # # # # # # # # #
            # remove idle segments  #
            # # # # # # # # # # # # #

            # get the null segments where the person is awaiting instructions
            inactive_frames = []
            f = open(null_segment, "r")
            for x in f:
              x = x.split("-")
              l = int(x[0])
              r = int(x[1])
              inactive_frames.extend(range(l,r+1))
            # get the non null segments
            active_frames = range(inactive_frames[-1]+1)
            active_frames = list(set(active_frames) - set(inactive_frames))

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # collect segments into 2 separate dictionaries based on action (pick/place)  #
            # and remove the first and last portions of each segment                      #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                                    
            # collect the pick and place segments
            for action in self.actions:
            
                valid_action_frames = []
                
                # read the segments for the current sequence
                segments = h5py.File(segment_file, 'r')['segments']
                for segment_x,segment_y in zip(segments[:-1], segments[1:]):
                    
                    # get the current segment
                    x_label = segment_x["label"].decode("utf-8")
                    y_label = segment_y["label"].decode("utf-8")
                    
                    # pick segment
                    if action == "pick" and x_label == "null" and y_label != "null":
                        #valid_action_frames.extend(range(segment_x[0],segment_x[1]))
                        valid_action_frames.extend(range(segment_x[0] + self.inp_length * self.time_step_size, segment_x[1] - self.out_length * self.time_step_size))

                    # place segment
                    if action == "place" and x_label != "null" and y_label == "null":
                        #valid_action_frames.extend(range(segment_x[0],segment_x[1]))
                        valid_action_frames.extend(range(segment_x[0] + self.inp_length * self.time_step_size, segment_x[1] - self.out_length * self.time_step_size))
                        
                
                # # # # # # # # # # # # #  
                # get the valid frames  # 
                # # # # # # # # # # # # #
                
                valid_frames = sorted(list(set(valid_action_frames) & set(active_frames) & set(valid_fixation_frames) & set(after_first_M_frames) & set(after_first_N_frames) & set(non_null_entries) & set(non_erroneous_frames)))
                
                pose_files      = [pose_files_all[i] for i in valid_frames]
                gaze_files      = [gaze_files_all[i] for i in valid_frames]
                object_files    = [object_files_all[i] for i in valid_frames]
                metadata_files  = [metadata_files_all[i] for i in valid_frames]    
                actions         = [action]*len(valid_frames)  
                fixation_angles = [fixation_all.iloc[i].values[0] for i in valid_frames]
                                
                # # # # # # # # # # # # # # # # # # # # #
                # get instructions for the valid frames #
                # # # # # # # # # # # # # # # # # # # # #  
                
                # read instruction file            
                #instruction_df = pd.read_csv(instruction_file, header=None, converters={3: literal_eval, 5: literal_eval})
                instruction_df= pd.read_excel(instruction_file, index_col=None)
                                
                # append initial and last frame
                instruction_segments = [0] + instruction_df.iloc[:,0].tolist() + [segments[-1][1]]
                
                # get the tuple (item,quantity,destination)
                instruction_tuples = instruction_df.iloc[:,2:]                                     
                instruction_tuples.columns = [0,1,2,3]
                instruction_tuples = pd.concat((pd.DataFrame([["NA",["NA"],"NA",["NA"]]]),instruction_tuples),axis=0,ignore_index=True)
                                
                # get the instructions at the valid_frames
                instructions = []
                for start,end,(index,row) in zip(instruction_segments[:-1],instruction_segments[1:],instruction_tuples.iterrows()):
                    instructions.extend([{"start":start, "end":end, "raw_text":row[0], "items":row[1], "quantity":row[2], "destinations":row[3]}]*(end-start))
                instructions = [instructions[i] for i in valid_frames]  
                assert not any([x["items"] == "NA" for x in instructions]) # should not have any "NA" instructions
                
                # sanity check
                #for i,instruction in enumerate(instructions):
                #    print(i, instruction)
                #sys.exit()
                
                # # # # # # # # # # # #               
                # put into dictionary #
                # # # # # # # # # # # #
                                         
                pose_list.extend(pose_files)
                gaze_list.extend(gaze_files)
                object_list.extend(object_files)
                metadata_list.extend(metadata_files)
                action_list.extend(actions)
                fixation_list.extend(fixation_angles)
                instruction_list.extend(instructions)
                
                max_pose_len.append(len(pose_files))
                print(dtype, action, "samples", len(pose_files), len(gaze_files), len(object_files), len(metadata_files), len(actions), len(instructions), len(fixation_angles), flush=True)
        
        print(dtype, "samples", len(pose_list), len(gaze_list), len(object_list), len(metadata_list), len(action_list), len(instruction_list), len(fixation_list), max(max_pose_len), flush=True)
        self.data = pd.DataFrame({"pose": pose_list, "gaze": gaze_list, "object": object_list, "metadata": metadata_list, "action":action_list, "fixation_angle":fixation_list, "instruction":instruction_list})  
                                                    
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
                                
        # filename
        data = self.data.iloc[idx]  
        
        # get metadata
        metadata       = data["metadata"]
        metadata       = pd.read_csv(metadata)
        action         = data["action"]
        fixation_angle = data["fixation_angle"]
        instruction    = data["instruction"]
        frame           = metadata["frame"].values[0]
        sequence        = metadata["sequence"].values[0]
        start_frame     = metadata["start_frame"].values[0]
        inp_frame       = metadata["frame"].values[0]
        key_frame       = metadata["key_frame"].values[0]
        key_object_name = metadata["key_object"].values[0]
        
        # # # # # # #
        # Pose Data #
        # # # # # # #
        
        # get inp pose
        inp_pose_filename = os.path.join(self.dataset_root,"xyz-poses",sequence,str(inp_frame).zfill(10)+".txt")
        inp_pose = load_pose(inp_pose_filename)
        center = inp_pose[11] 
                
        # get key pose
        key_pose_filename = os.path.join(self.dataset_root,"xyz-poses",sequence,str(key_frame).zfill(10)+".txt")
        key_pose = load_pose(key_pose_filename)

        # # # # # # #
        # Eye Gaze  #
        # # # # # # #

        # get the eye gazes
        gaze_vector = []
        for i in range(inp_frame-(self.gaze_length-1)*self.time_step_size, inp_frame+1, self.time_step_size):
            gaze_filename = os.path.join(self.dataset_root,"eye-gaze-processed",sequence,str(i).zfill(10)+".csv")
            gaze_vector.append(load_gaze(gaze_filename))
        gaze_vector = np.array(gaze_vector)  # [inp_length, 2, 3]
        
        # # # # # # # # # # # # # # # # # # # # #
        # pick action - get object coordinates  #
        # # # # # # # # # # # # # # # # # # # # #
        
        # - provide label for each object for whether or not to process (No. Instead, just remove them from the list)
        if action == "pick":
        
            # read the object file
            objects_filename = os.path.join(self.dataset_root,"object-positions-orientations",sequence,str(key_frame).zfill(10)+".txt")
            
            # get all objects
            all_objects = pd.read_csv(objects_filename, sep=" ")
            
            # get the key object
            key_object = all_objects[all_objects["name"] == key_object_name]
            key_object = key_object[["x","y","z"]].values
                  
            # remove non interactable objects
            objects = all_objects[~all_objects['name'].str.contains("shelf") & ~all_objects['name'].str.contains("table") & ~all_objects['name'].str.contains("chair")]#.reset_index(drop=True)
                                    
            # if using instructions
            if self.use_instructions:
                
                # # # # # # # # # # # # # #
                # set table for X persons #
                # # # # # # # # # # # # # #
                
                if literal_eval(instruction["destinations"]) == ["table"]:
                    
                    # X persons where X = quantity
                    quantity = instruction["quantity"]
                    
                    # # # # # # # # # # # # # # # # # # # # # # # # # # #
                    # Compute the total number of objects at the table  #
                    # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                    
                    # get destinations
                    destinations = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table")]
                    
                    # get the total number of each object at the table
                    object_count = {"cup":0, "plate":0, "bowl":0, "jug":0}
                    for index, row in objects.iterrows():
                        
                        name = get_noun(row["name"])
                        coords = np.array([row["x"],row["y"],row["z"]])
                        if where(coords, destinations) == "table":
                            object_count[name] = object_count[name]+1
                                
                    # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                    # Determine if each object needs to be moved (picked) #
                    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                    
                    remove = []                    
                    for index, row in objects.iterrows():
                        
                        sanity_check = 0
                        name = get_noun(row["name"])
                        coords = np.array([row["x"],row["y"],row["z"]])
                        
                        # case 1: do not add if object not at destination and count >= quantity
                        if where(coords, destinations) != "table" and object_count[name] >= quantity:
                            sanity_check += 1
                            remove.append(index)
                            
                        # case 2: add if object not at destination and count < quantity
                        if where(coords, destinations) != "table" and object_count[name] < quantity:
                            sanity_check += 1
                            pass
                        
                        # case 4: do not add if object at destination and count <= quantity
                        if where(coords, destinations) == "table" and object_count[name] <= quantity:
                            sanity_check += 1
                            remove.append(index)
                        
                        # case 3: add if object at destination and count > quantity    
                        if where(coords, destinations) == "table" and object_count[name] > quantity:
                            sanity_check += 1
                            pass
                    
                    # remove objects
                    objects = objects.drop(index=remove)

                # # # # # # # # # # # # # # # # # # # #
                # put jug and bowl on shelf           #
                # put blue and green objects on shelf #
                # put all cups on shelf               #
                # put all plates on shelf             #
                # # # # # # # # # # # # # # # # # # # #
                                                
                else:
                                   
                    # remove objects not in the instruction
                    objects = objects[objects["name"].str.contains("|".join(literal_eval(instruction["items"])))]
                    
                    # get destinations
                    destinations = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table")]
                        
                    # remove objects that are already at the destination    
                    remove = []
                    for index,row in objects.iterrows():
                    
                        coords = np.array([row["x"],row["y"],row["z"]])
                        if where(coords, destinations) in literal_eval(instruction["destinations"]):
                            remove.append(index)
                            
                    objects = objects.drop(index=remove)
            
            #print(sequence, frame, instruction["raw_text"])
            #print("start_frame", start_frame, "key_frame", key_frame, action)
            #print(objects)
            #print(key_object_name)
            #print()
             
            # get their names, score and coordinates
            object_names = objects["name"].values
            object_scores = objects.index[objects["name"] == key_object_name].values[0] # this confirms that the coordinates being passed to the model contains the key object as well
            object_indexes = np.array(objects.index)
            object_coordinates = objects[["x","y","z"]].values # [num_objects, 3]
        
        # # # # # # # # # # # # # # # # # # # # # #
        # place action - create grid coordinates  #
        # # # # # # # # # # # # # # # # # # # # # #
        
        if action == "place":
                
            # no id for place action
            object_scores = np.int64(-1)
            
            # read the object file at the start of the segment
            # since the object will end up being too close to the destination if i use the current frame (WRONG, because i read the key object here too)
            objects_filename = os.path.join(self.dataset_root,"object-positions-orientations",sequence,str(key_frame).zfill(10)+".txt")
            
            # get all objects
            all_objects = pd.read_csv(objects_filename, sep=" ")
            
            # get the key object
            key_object = all_objects[all_objects["name"] == key_object_name]
            key_object = key_object[["x","y","z"]].values   
            
            # get furnitures
            objects = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table")]
            
            if self.use_instructions:
                            
                objects = objects[objects["name"].str.contains(where(key_object,objects))]    
                
                """# # # # # # # # # # # # # #
                # set table for X persons #
                # # # # # # # # # # # # # #
                    
                if literal_eval(instruction["destinations"]) == ["table"]:
                
                    # X persons where X = quantity
                    quantity = instruction["quantity"]
                    
                    # # # # # # # # # # # # # # # # # # # # # # # # # # #
                    # Compute the total number of objects at the table  #
                    # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                    
                    inp_objects_filename = os.path.join(self.dataset_root,"object-position-orientation-files",self.dtype,sequence,str(start_frame).zfill(10)+".txt")
                    inp_objects = pd.read_csv(inp_objects_filename, sep=" ")
                    inp_objects = inp_objects[~inp_objects['name'].str.contains("shelf") & ~inp_objects['name'].str.contains("table") & ~inp_objects['name'].str.contains("chair")]
                                        
                    # get the total number of each object at the table
                    object_count = {"cup":0, "plate":0, "bowl":0, "jug":0}
                    for index, row in inp_objects.iterrows():
                        
                        name = get_noun(row["name"])
                        coords = np.array([row["x"],row["y"],row["z"]])
                        if where(coords, objects) == "table":
                            object_count[name] = object_count[name]+1     
                            
                    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                    # Determine if the key object will be placed at the table or away from it #
                    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                                        
                    # get the key object count at the table
                    key_object_count = object_count[get_noun(key_object_name)]
                
                    # case 1: person is putting the object elsewhere if key_object_count >= quantity
                    if key_object_count >= quantity:
                        objects = all_objects[all_objects["name"].str.contains("shelf")]
                        
                    # case 2: person is putting the object on the table if key_object_count < quantity
                    if key_object_count <= quantity:
                        objects = all_objects[all_objects["name"].str.contains("table")]
                
                # # # # # # # # # # # # # # # # # # # #
                # put jug and bowl on shelf           #
                # put blue and green objects on shelf #
                # put all cups on shelf               #
                # put all plates on shelf             #
                # # # # # # # # # # # # # # # # # # # #
                                                
                else:
                
                    # remove objects not in the instruction
                    objects = objects[objects["name"].str.contains("|".join(literal_eval(instruction["destinations"])))]"""
                
            # # # # # # # # # # # # # # # # # # # # # # #
            # create a grid of the selected furnitures  #
            # # # # # # # # # # # # # # # # # # # # # # #
                
            object_indexes, object_names, object_coordinates = [], [], []    
            for index,row in objects.iterrows():
                
                # object name
                object_name = row["name"]
                
                # get furniture bounding box at origin
                object_bbox = pd.read_csv(os.path.join(self.dataset_root,"bboxes",object_name+".csv"))
                
                # get the grid parameters
                translation = row[["x","y","z"]].values
                rotation = row[["a","b","c","d"]].values
                grid_size = self.grid_sizes[self.furniture_names.index(object_name)]
                
                # generate grid
                grid = generate_grid(object_bbox, translation, rotation, grid_size[0], grid_size[1], object_name)
                object_coordinates.append(grid)
                object_names.append(object_name)
                object_indexes.append(index)
                
            object_coordinates = np.concatenate(object_coordinates).astype(np.float32)
            object_indexes = np.array(object_indexes)
            
            #print(sequence, frame, instruction["raw_text"])
            #print("start_frame", start_frame, "key_frame", key_frame, action, key_object_name)
            #print(where(key_object, all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table")]))
            #print(object_names)
            #input()
            #assert where(key_object, all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table")]) in object_names
        
        # do not delete, sanity check
        # for pick action, key_object_location is where the object will be picked from
        # for place action, key_object_location is where the object was originally from
        
        """if action == "pick":    
            furnitures = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table")]
            key_object_location = where(key_object, furnitures)     
            print(sequence, frame)
            print("instruction",instruction)
            print("action",action)
            print("all_objects",all_objects["name"].tolist())
            print("objects_in_use",object_names)
            print("key_object",key_object_name)
            print("where the object will be picked from",key_object_location)
            print("objects_in_use_indexes",object_indexes)
            print(object_coordinates.shape)
            print()
            print()
        if action == "place":       
            inp_objects_filename = os.path.join(self.dataset_root,"object-position-orientation-files",self.dtype,sequence,str(instruction["start"]).zfill(10)+".txt")
            inp_objects = pd.read_csv(inp_objects_filename, sep=" ")
            inp_objects = inp_objects[~inp_objects['name'].str.contains("shelf") & ~inp_objects['name'].str.contains("table") & ~inp_objects['name'].str.contains("chair")]
            inp_key_object = inp_objects[inp_objects["name"] == key_object_name]
            inp_key_object = inp_key_object[["x","y","z"]].values 
            furnitures = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table")]
            key_object_location = where(inp_key_object, furnitures) 
            print(sequence, frame)
            print("instruction",instruction)
            print("action",action)
            print("all_objects",all_objects["name"].tolist())
            print("objects_in_use",object_names)
            print("key_object",key_object_name)
            print("where the object was originally from",key_object_location)
            print("objects_in_use_indexes",object_indexes)
            print(object_coordinates.shape)
            print()
            print()
        input()"""
                                
        # # # # # # # # # #       
        # preprocess data #
        # # # # # # # # # #
                
        # normalize gaze vector
        # gaze vector is the person's eye gaze
        normalized_gaze_vector = gaze_vector - center                                      # [gaze length, 2, 3]
        normalized_gaze_vector = normalized_gaze_vector[:,1] - normalized_gaze_vector[:,0] # [gaze length, 3]
        normalized_gaze_vector = normalized_gaze_vector / np.linalg.norm(normalized_gaze_vector, axis=-1, keepdims=True) # [gaze length, 3]
                
        # normalize object vectors 
        # object vector points from the person's eye to the object
        object_vector = []
        for x in object_coordinates:
            # g[0] is the coordinates of the head / eye
            # x is the coordinates of the object
            object_vector.append(np.array([[g[0],x] for g in gaze_vector])) # [gaze length, 2 ,3] 
        object_vector = np.array(object_vector)                             # [10, gaze length, 2 ,3]
        normalized_object_vector = object_vector - center                   # [10, gaze length, 2 ,3]
        normalized_object_vector = normalized_object_vector[:,:,1] - normalized_object_vector[:,:,0]                           # [10, gaze length, 3]
        normalized_object_vector = normalized_object_vector / np.linalg.norm(normalized_object_vector, axis=-1, keepdims=True) # [10, gaze length, 3]
        
        # get the relative vector object - gaze
        relative_gaze = normalized_object_vector - normalized_gaze_vector # [10, gaze length, 3]
        key_relative_gaze = np.median(relative_gaze,axis=1)               # [10, 3]
        relative_gaze = relative_gaze.astype(np.float32)
        key_relative_gaze = key_relative_gaze.astype(np.float32) 
        
        # subtract the x,y,z coordinates of all by the x,y,z coordinates of the hip
        gaze_vector = gaze_vector - center
        inp_pose = inp_pose - center
        key_pose = key_pose - center
        key_object = (key_object - center).astype(np.float32)
        object_scores = object_scores.astype(np.long)
        object_coordinates = (object_coordinates - center).astype(np.float32)
                
        # # # # # # # # # # # # # # # # # # # # # # # # #
        # pad relative_gaze, key_relative_gaze, objects #
        # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # pad relative gaze
        relative_gaze_pad = np.zeros((self.object_padded_length,relative_gaze.shape[1],relative_gaze.shape[2]),dtype=relative_gaze.dtype)
        relative_gaze_pad[:relative_gaze.shape[0],:relative_gaze.shape[1],:relative_gaze.shape[2]] = relative_gaze
              
        # pad key relative gaze  
        key_relative_gaze_pad = np.zeros((self.object_padded_length,key_relative_gaze.shape[1]),dtype=key_relative_gaze.dtype)
        key_relative_gaze_pad[:key_relative_gaze.shape[0],:key_relative_gaze.shape[1]] = key_relative_gaze
                
        # pad object coordinates
        object_coordinates_padded = np.zeros((self.object_padded_length,object_coordinates.shape[1]),dtype=object_coordinates.dtype)
        object_coordinates_padded[:object_coordinates.shape[0],:object_coordinates.shape[1]] = object_coordinates
        
        # pad objects in use
        object_indexes = np.array(object_indexes)
        object_indexes_padded = np.zeros((self.object_padded_length),dtype=int)
        object_indexes_padded[:object_indexes.shape[0]] = object_indexes
        
        # keep unpadded length
        relative_gaze_unpadded_length       = relative_gaze.shape[0]
        key_relative_gaze_unpadded_length   = key_relative_gaze.shape[0]
        object_coordinates_unpadded_length  = object_coordinates.shape[0]
        object_indexes_unpadded_length      = object_indexes.shape[0]
                
                # metadata
        data = {"idx":idx, "sequence":sequence,  "filename":os.path.join(sequence,str(frame).zfill(10)),
        
                # frame data
                "inp_frame":inp_frame, "key_frame":key_frame, 
        
                # pose
                "inp_pose":inp_pose, "key_pose":key_pose, 
                
                # action
                "instruction": instruction["raw_text"], "action":action,
        
                # gaze
                "center":center, "gaze_vector":gaze_vector, 
                "relative_gaze":relative_gaze_pad, "relative_gaze_unpadded_length":relative_gaze_unpadded_length, 
                "key_relative_gaze":key_relative_gaze_pad, "key_relative_gaze_unpadded_length":key_relative_gaze_unpadded_length,
                "fixation_angle":fixation_angle, 
                                
                # object
                "object_scores":object_scores, "key_object":key_object, "key_object_name":key_object_name, 
                "object_indexes":object_indexes_padded, "object_indexes_unpadded_length":object_indexes_unpadded_length,
                "objects":object_coordinates_padded, "objects_unpadded_length":object_coordinates_unpadded_length,
                
                # pad data
                "object_padded_length":self.object_padded_length, "unpadded_length":relative_gaze_unpadded_length
                }
                        
        return data

def get_noun(full_name):
    nouns = ["cup","plate","bowl","jug","laiva_shelf"]
    for noun in nouns:
        if noun in full_name:
            return noun

# determine which furniture the current object is in
def where(object_coords, furniture):

    min_index, min_dist = np.inf, np.inf
    for index,row in furniture.iterrows():
    
        # compute distance to each furniture
        furniture_coords = row[["x","y","z"]].values
        dist = np.sqrt(np.sum((object_coords - furniture_coords)**2))
        min_index, min_dist = (index, dist) if dist < min_dist else (min_index, min_dist)
    
    # return name of closest furniture
    return furniture.loc[min_index]["name"]

# generate the grid for the furniture
def generate_grid(bbox, translation, rotation, nx, ny, name):

    assert name == "table" or name == "laiva_shelf" or name == "vesken_shelf"

    min_x, max_x = bbox["min_x"].values[0], bbox["max_x"].values[0]
    min_y, max_y = bbox["min_y"].values[0], bbox["max_y"].values[0]
    min_z, max_z = bbox["min_z"].values[0], bbox["max_z"].values[0]
    
    if name == "table":
        
        # shift the min and max so i dont get a grid at the edge of the table
        len_x = max_x - min_x
        min_x = min_x + 0.1*len_x
        max_x = max_x - 0.1*len_x
        
        len_y = max_y - min_y
        min_y = min_y + 0.1*len_y
        max_y = max_y - 0.1*len_y
        
        if ny == 1:
            min_y = max_y = (max_y + min_y)/2
        
        # generate the 1D grids
        x = np.linspace(min_x, max_x, nx)
        y = np.linspace(min_y, max_y, ny)
        z = [max_z]        
        
    if name == "laiva_shelf":
        
        # shift the min and max so i dont get a grid at the edge of the shelf
        len_x = max_x - min_x
        min_x = min_x + 0.2*len_x
        max_x = max_x - 0.2*len_x

        len_y = max_y - min_y
        min_y = min_y + 0.25*len_y
        max_y = max_y - 0.25*len_y        
        
        if ny == 1:
            min_y = max_y = (max_y + min_y)/2
               
        # generate the 1D grids
        x = np.linspace(min_x, max_x, nx)
        y = np.linspace(min_y, max_y, ny)
        z = np.linspace(-0.690,0.715,5)
    
    if name == "vesken_shelf":
        
        # shift the min and max so i dont get a grid at the edge of the shelf
        len_x = max_x - min_x
        min_x = min_x + 0.2*len_x
        max_x = max_x - 0.2*len_x
        
        len_y = max_y - min_y
        min_y = min_y + 0.25*len_y
        max_y = max_y - 0.25*len_y
        
        if ny == 1:
            min_y = max_y = (max_y + min_y)/2
        
        # generate the 1D grids
        x = np.linspace(min_x, max_x, nx)
        y = np.linspace(min_y, max_y, ny)
        z = np.linspace(-0.45,0.45,4)
            
    # create the 3D grid
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))
    grid = np.stack((X,Y,Z),axis=1)
        
    # transform the 3D grid
    grid = transform_object(grid, translation=translation, rotation=rotation)
    
    return grid

# transform point clouds    
def transform_object(vertices, translation, rotation):
       
    translation=np.squeeze(translation)
    rotation=np.squeeze(rotation)
          
    # get rotation matrix
    rotation = R.from_quat(rotation).as_matrix()
        
    # rotate about origin first then translate
    vertices = np.matmul(rotation,vertices.T).T + translation
    return vertices 
