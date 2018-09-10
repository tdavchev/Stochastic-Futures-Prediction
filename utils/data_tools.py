import os
import random
import numpy as np

def mil_to_pixels(directory=["./data/ewap_dataset/seq_hotel"]):
    '''
    Preprocess the frames from the datasets.
    Convert values to pixel locations from millimeters
    obtain and store all frames data the actually used frames (as some are skipped), 
    the ids of all pedestrians that are present at each of those frames and the sufficient statistics.
    '''
    def collect_stats(agents):
        x_pos = []
        y_pos = []
        for agent_id in range(1, len(agents)):
            trajectory = [[] for _ in range(3)]
            traj = agents[agent_id]
            for step in traj:
                x_pos.append(step[1])
                y_pos.append(step[2])
        x_pos = np.asarray(x_pos)
        y_pos = np.asarray(y_pos)
        # takes the average over all points through all agents
        return [[np.min(x_pos), np.max(x_pos)], [np.min(y_pos), np.max(y_pos)]]

    Hfile = os.path.join(directory, "H.txt")
    obsfile = os.path.join(directory, "obsmat.txt")
    # Parse homography matrix.
    H = np.loadtxt(Hfile)
    Hinv = np.linalg.inv(H)
    # Parse pedestrian annotations.
    frames, pedsInFrame, agents = parse_annotations(Hinv, obsfile)
    # collect mean and std
    statistics = collect_stats(agents)
    norm_agents = []
    # collect the id, normalised x and normalised y of each agent's position
    pedsWithPos = []
    for agent in agents:
        norm_traj = []
        for step in agent:
            _x = (step[1] - statistics[0][0]) / (statistics[0][1] - statistics[0][0])
            _y = (step[2] - statistics[1][0]) / (statistics[1][1] - statistics[1][0])
            norm_traj.append([int(frames[int(step[0])]), _x, _y])

        norm_agents.append(np.array(norm_traj))

    return np.array(norm_agents), statistics, pedsInFrame

def parse_annotations(Hinv, obsmat_txt):
    '''
    Parse the dataset and convert to image frames data.
    '''
    def to_image_frame(loc):
        """
        Given H^-1 and (x, y, z) in world coordinates, 
        returns (u, v, 1) in image frame coordinates.
        """
        loc = np.dot(Hinv, loc) # to camera frame
        return loc/loc[2] # to pixels (from millimeters)

    mat = np.loadtxt(obsmat_txt)
    num_peds = int(np.max(mat[:,1])) + 1
    peds = [np.array([]).reshape(0,4) for _ in range(num_peds)] # maps ped ID -> (t,x,y,z) path
    
    num_frames = (mat[-1,0] + 1).astype("int")
    num_unique_frames = np.unique(mat[:,0]).size
    recorded_frames = [-1] * num_unique_frames # maps timestep -> (first) frame
    peds_in_frame = [[] for _ in range(num_unique_frames)] # maps timestep -> ped IDs

    frame = 0
    time = -1
    blqk = False
    for row in mat:
        if row[0] != frame:
            frame = int(row[0])
            time += 1
            recorded_frames[time] = frame

        ped = int(row[1])

        peds_in_frame[time].append(ped)
        loc = np.array([row[2], row[4], 1])
        loc = to_image_frame(loc)
        loc = [time, loc[0], loc[1], loc[2]]
        peds[ped] = np.vstack((peds[ped], loc))

    return recorded_frames, peds_in_frame, peds

def preprocess(data_dirs):
    '''
    Collect all data from the already preprocessed csv files.
    These files are used for training and to speed up the process
    of training as opposed to loading and preprocessing each and every
    one individually.
    modified from: https://github.com/vvanirudh/social-lstm-tf
    '''
    all_ped_data = []
    all_ped_dict = {}
    dataset_indices = []
    current_ped = 0
    for directory in data_dirs:
        file_path = os.path.join(directory, 'pixel_pos.csv')
        data = np.genfromtxt(file_path, delimiter=',')

        numPeds = np.size(np.unique(data[1, :]))

        for ped in range(1, numPeds+1):
            traj = data[:, data[1, :] == ped]
            traj = traj[[0, 3, 2], :].T

            all_ped_data.append(np.array(traj))
            all_ped_dict[current_ped + ped] = traj

        dataset_indices.append(current_ped+numPeds)
        current_ped += numPeds

    return np.array(all_ped_data), all_ped_dict, dataset_indices

def load_preprocessed(pedData, batch_size, sequence_length):
    '''
    Function to load the pre-processed data into the DataLoader object
    params:
    data_file : The path to the pickled data file
    '''
    # Construct the data with sequences(or trajectories) longer than sequence_length
    load_data = []
    counter = 0
    # For each pedestrian in the data
    for idx, traj in enumerate(pedData):
        # Extract their trajectory
        # If the length of the trajectory is greater than sequence_length
        # (+1 as we need both source and target data)
        if traj.shape[0] > (sequence_length+2):
            load_data.append(traj)
            # Number of batches this datapoint is worth
            counter += int(traj.shape[0] / ((sequence_length+2)))

    # Calculate the number of batches (each of batch_size) in the data
    num_batches = int(counter / batch_size)

    return load_data, num_batches

def tick_batch_pointer(pointer, data_len):
    # Advance the data pointer
    pointer += 1
    if (pointer >= data_len):
        return  0

    return pointer

def reset_batch_pointer():
    # Reset the data pointer
    return 0

def next_batch(_data, pointer, batch_size, sequence_length, infer=False):
    '''
    Function to get the next batch of points
    '''
    # List of source and target data for the current batch
    x_batch = []
    y_batch = []
    # For each sequence in the batch
    for i in range(batch_size):
        # Extract the trajectory of the pedestrian pointed out by self.pointer
        traj = _data[pointer]
        # Number of sequences corresponding to his trajectory
        n_batch = int(traj.shape[0] / (sequence_length+1))
        # Randomly sample an index from which his trajectory is to be considered
        if not infer:
            idx = random.randint(0, traj.shape[0] - sequence_length - 1)
        else:
            idx = 0

        # Append the trajectory from idx until sequence_length into source and target data
        x_batch.append(np.copy(traj[idx:idx+sequence_length, :]))
        y_batch.append(np.copy(traj[idx+1:idx+sequence_length+1, :]))

        if random.random() < (1.0/float(n_batch)):
            # Adjust sampling probability
            # if this is a long datapoint, sample this data more with
            # higher probability
            pointer = tick_batch_pointer(pointer, len(_data))

    return x_batch, y_batch, pointer