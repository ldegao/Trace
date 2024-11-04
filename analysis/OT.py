import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean
from scipy.sparse import dok_matrix
from sklearn.manifold import MDS


# Load JSON file
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}, skipping this file.")
    except FileNotFoundError:
        print(f"File not found: {file_path}, skipping this file.")
    except Exception as e:
        print(f"Unexpected error reading file {file_path}: {e}, skipping this file.")
    return None


# Extract frames from the data, treating the player as a regular NPC
def extract_frames(data):
    frames = []
    for time_step, details in sorted(data.items()):
        if not isinstance(details, dict) or not details.get("player"):
            continue

        frame = {'npcs': []}
        player_position = details["player"]["transform"]["location"]
        player_velocity = details["player"]["velocity"]
        player_data = {
            'position': np.array([player_position['x'], player_position['y'], player_position['z']]),
            'velocity': np.array([player_velocity['x'], player_velocity['y'], player_velocity['z']]),
            'type': 'player'
        }
        frame['npcs'].append(player_data)

        if "NPC" in details and isinstance(details["NPC"], list):
            for npc in details["NPC"]:
                npc_position = npc["transform"]["location"]
                npc_velocity = npc["velocity"]
                npc_data = {
                    'position': np.array([npc_position['x'], npc_position['y'], npc_position['z']]),
                    'velocity': np.array([npc_velocity['x'], npc_velocity['y'], npc_velocity['z']]),
                    'type': 'npc'
                }
                frame['npcs'].append(npc_data)

        frames.append(frame)
    return frames


# Calculate the state distance between two nodes (vehicles)
def node_distance(node1, node2):
    pos_dist = euclidean(node1['position'], node2['position'])
    vel_dist = euclidean(node1['velocity'], node2['velocity'])
    return pos_dist + vel_dist


# Compute optimal transport distance without a threshold mechanism
def compute_ot_distance_pruned(npcs1, npcs2, k=5):
    num_nodes1, num_nodes2 = len(npcs1), len(npcs2)
    cost_matrix = dok_matrix((num_nodes1, num_nodes2))

    for i in range(num_nodes1):
        distances = [(j, node_distance(npcs1[i], npcs2[j])) for j in range(num_nodes2)]
        distances = sorted(distances, key=lambda x: x[1])[:k]
        for j, dist in distances:
            cost_matrix[i, j] = dist

    cost_matrix_dense = cost_matrix.toarray()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_dense)
    total_cost = cost_matrix_dense[row_ind, col_ind].sum()

    return total_cost


# Graph-DTW for alignment
def graph_dtw(sequence1, sequence2, k=5):
    len1, len2 = len(sequence1), len(sequence2)
    dtw_matrix = np.full((len1 + 1, len2 + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = compute_ot_distance_pruned(sequence1[i - 1]['npcs'], sequence2[j - 1]['npcs'], k=k)
            min_path = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + min_path

    return dtw_matrix[len1, len2]


# Sampling function to reduce the number of frames
def sample_sequence(sequence, sampling_rate=10):
    return sequence[::sampling_rate]


import os
import json
import numpy as np


def calculate_similarity_matrix(base_folder_path, sampling_rate=10):
    file_paths = []
    data_list = []
    min_dists = []

    # ?? JSON ??
    for root, dirs, files in os.walk(base_folder_path):
        for filename in files:
            if filename.endswith('.json') and 'time_record' in root:
                file_path = os.path.join(root, filename)
                data = load_json(file_path)
                if data is None:
                    continue

                min_dists.append(data.get("min_dist", 0.5))
                extracted_frames = extract_frames(data)
                sampled_frames = sample_sequence(extracted_frames, sampling_rate=sampling_rate)
                data_list.append(sampled_frames)
                file_paths.append(file_path)

    num_files = len(file_paths)
    similarity_matrix = np.zeros((num_files, num_files))

    for i in range(num_files):
        for j in range(i, num_files):
            if i == j:
                similarity_matrix[i, j] = 0
            else:
                print(f"Calculating similarity between file {i + 1}/{num_files} and file {j + 1}/{num_files}")
                try:
                    score = graph_dtw(data_list[i], data_list[j])
                    similarity_matrix[i, j] = similarity_matrix[j, i] = score
                    print(f"Similarity score between file {i} and file {j}: {score}")
                except Exception as e:
                    print(f"Error calculating similarity between file {i} and file {j}: {e}")
                    similarity_matrix[i, j] = similarity_matrix[j, i] = np.inf  # ????????? inf

    large_constant = np.max(similarity_matrix[np.isfinite(similarity_matrix)]) * 1.1
    similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0, posinf=large_constant)

    np.save('similarity_matrix.npy', similarity_matrix)
    return similarity_matrix, min_dists, file_paths


# Plot using MDS
def plot_similarity_matrix(similarity_matrix, min_dists, file_paths):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    points = mds.fit_transform(similarity_matrix)

    min_dists_normalized = (min_dists - np.min(min_dists)) / (np.max(min_dists) - np.min(min_dists))

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=min_dists_normalized, cmap='coolwarm', s=100, edgecolor='k')
    plt.colorbar(scatter, label="min_dist")

    for i, txt in enumerate(file_paths):
        plt.annotate(os.path.basename(txt), (points[i, 0], points[i, 1]))

    plt.xlabel("MDS: X")
    plt.ylabel("MDS: Y")
    plt.title("Similarity of Scenarios")
    plt.show()


# Main function
def main(folder_path, sampling_rate=24):
    similarity_matrix, min_dists, file_paths = calculate_similarity_matrix(folder_path, sampling_rate=sampling_rate)
    plot_similarity_matrix(similarity_matrix, min_dists, file_paths)


# Execute main function
if __name__ == "__main__":
    folder_path = '../data/save'  # Replace with the path to your JSON files folder
    main(folder_path)
