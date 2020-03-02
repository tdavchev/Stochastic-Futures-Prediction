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

def preprocess_frames(data_dirs, max_num_agents=40):
    '''
    Collect all data from the already preprocessed csv files.
    These files are used for training and to speed up the process
    of training as opposed to loading and preprocessing each and every
    one individually.
    The spit out format is frame oriented where each frame has a list of
    agents and their locations.
    NOTE: modified from https://github.com/vvanirudh/social-lstm-tf
    args:
        data_dirs                : the directories names used
    returns:
        all_frames_data          : an np.array() of size [num_dirs x num_unique_frames x max_num_agents x 3 (aid, x, y)]
        frames_list              : an array with the actual frames ordered accordingly
        dataset_endframe_indices : the indices of the last frame per dataset
    '''
    frames_list = []
    dataset_endframe_indices = []
    current_ped = 0
    all_frames_data = []
    for idx, directory in enumerate(data_dirs):

        file_path = os.path.join(directory, 'pixel_pos.csv')
        data = np.genfromtxt(file_path, delimiter=',')

        # num_peds = np.size(np.unique(data[1, :]))
        unique_frames = np.unique(data[0, :])
        frames_list.append(unique_frames)
        dataset_endframe_indices.append(len(unique_frames))
        all_frames_data.append(np.zeros((len(unique_frames), max_num_agents, 3)))
        for f_id, frame in enumerate(unique_frames):
            agents_in_frame = data[:, data[0, :] == frame]
            aids_xs_ys = agents_in_frame[[1, 2, 3], :].T
            all_frames_data[idx][f_id, 0:len(aids_xs_ys), :] = np.array(aids_xs_ys)

    return np.array(all_frames_data), frames_list, dataset_endframe_indices

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

def load_preprocessed_frames(frame_data, batch_size, sequence_length):
    '''
    Function to load the pre-processed data in terms of [frame, agent, x, y]
    params:
    data_file : The path to the pickled data file
    '''
    # Construct the data with sequences(or trajectories) longer than sequence_length
    counter = 0
    for idx, cur_dset_data in enumerate(frame_data):
        print(int(len(cur_dset_data) / (sequence_length+1)))
        counter += int(len(cur_dset_data) / (sequence_length+1)) # not sure about 1 or 2 ...

    # Calculate the number of batches (each of batch_size) in the data
    num_batches = int(counter / batch_size)
    # On an average, we need twice the number of batches to cover the data
    # due to randomization introduced: https://github.com/vvanirudh/social-lstm-tf
    num_batches = num_batches * 2

    return num_batches

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

def tick_dataset_pointer(dataset_pointer, data_len):
    dataset_pointer += 1
    pointer = 0
    if dataset_pointer >= data_len:
        dataset_pointer = 0

    return dataset_pointer, pointer


def advance_frame_pointer(pointer, value):
    pointer += value
    return pointer

def reset_batch_pointer():
    # Reset the data pointer
    return 0

def reset_data_set_pointer():
    # Reset the data pointer
    return 0

def next_batch_frame(_data, _frames_list, pointer, dataset_pointer, batch_size, sequence_length, dataset_names, max_num_peds, rnd=None, random_update=True, infer=False):
    '''
    Function to get the next batch of points.
    TODO: Needs to fix the hardcoded dataset names ...
    '''
    # List of source and target data for the current batch
    x_batch = []
    y_batch = []
    d_batch = []
    f_batch = []
    # For each sequence in the batch
    i = 0
    while i < batch_size:
        frame_data = _data[dataset_pointer]
        frame_ids = _frames_list[dataset_pointer]

        idx = pointer
        if idx + sequence_length < frame_data.shape[0]:
            i += 1
            # All the data in this sequence
            seq_frame_data = frame_data[idx:idx+sequence_length+1, :]
            seq_source_frame_data = frame_data[idx:idx+sequence_length, :]
            seq_target_frame_data = frame_data[idx+1:idx+sequence_length+1, :]
            # Number of unique agents in this sequence of frames
            agents_id_list = np.unique(seq_frame_data[:, :, 0])
            num_unique_agents = agents_id_list.shape[0]

            input_data = np.zeros((sequence_length, max_num_peds, 3))
            target_data = np.zeros((sequence_length, max_num_peds, 3))
            fid_data = frame_ids[idx:idx+sequence_length] # I only need the input frames!

            for seq in range(sequence_length):
                iseq_frame_data = seq_source_frame_data[seq, :]
                tseq_frame_data = seq_target_frame_data[seq, :]
                for age, agent_id in enumerate(agents_id_list):#range(num_unique_agents):
                    # this should be enumerate over all agents_id_lists...!!!
                    # agent_id = agents_id_list[age]
                    if agent_id == 0:
                        continue
                    else:
                        iage = iseq_frame_data[iseq_frame_data[:, 0] == agent_id, :]
                        tage = np.squeeze(tseq_frame_data[tseq_frame_data[:, 0] == agent_id, :])
                        if iage.size != 0:
                            input_data[seq, age, :] = iage
                        if tage.size != 0:
                            target_data[seq, age, :] = tage

            x_batch.append(input_data)
            y_batch.append(target_data)
            f_batch.append(fid_data)

            if random_update:
                pointer = advance_frame_pointer(pointer, rnd.randint(1, sequence_length))
            else:
                pointer = advance_frame_pointer(pointer, 1)

            d_batch.append(dataset_names[dataset_pointer])
        else:
            dataset_pointer, pointer = tick_dataset_pointer(dataset_pointer, len(_data))

    return x_batch, y_batch, d_batch, f_batch, pointer, dataset_pointer


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