import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pickle


def gather_data(base_folder_path):
    """
    Gathers data from all JSON files in subdirectories of the base folder that contain 'time_record' in their path.

    Parameters:
    - base_folder_path: str, path to the base folder containing subdirectories with JSON files.

    Returns:
    - DataFrame with all gathered features.
    """
    # Initialize an empty list to store the averaged features for each JSON file
    data = []
    for root, dirs, files in os.walk(base_folder_path):
        dirs.sort()
        files.sort()
        for filename in files:
            if 'error' in root:
                continue
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                print(f"Processing file: {filepath}")

                # Calculate averaged features for each JSON file
                avg_features = get_average_features_for_json(filepath)

                if avg_features is not None:
                    data.append(avg_features)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)

    # Drop rows with NaN or inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"Total number of data points in the DataFrame: {len(df)}")
    return df


def get_average_features_for_json(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {filepath}, skipping this file.")
        return None

    total_features = {
        "filename": filepath,
        "avg_speed": 0.0,
        "avg_acceleration": 0.0,
        "min_x_dist": 0.0,
        "min_y_dist": 0.0,
        "ttc": 0.0
    }
    count = 0

    for timestamp, value in data.items():
        if timestamp in ["min_dist_frame", "min_dist"]:
            continue

        for entity_key in ["player", "NPC"]:
            entity_data = value.get(entity_key, {})
            if isinstance(entity_data, list):
                for npc in entity_data:
                    if add_entity_data_to_totals(npc, total_features):
                        count += 1
            elif isinstance(entity_data, dict):
                if add_entity_data_to_totals(entity_data, total_features):
                    count += 1

    if count > 0:
        for key in total_features:
            if key != "filename":
                total_features[key] /= count
            else:
                total_features[key] = filepath
        return total_features
    else:
        return None


def add_entity_data_to_totals(entity, total_features):
    velocity = entity.get("velocity", {})
    vx, vy, vz = velocity.get("x", 0), velocity.get("y", 0), velocity.get("z", 0)

    if np.isnan(vx) or np.isnan(vy) or np.isnan(vz):
        return False

    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    total_features["avg_speed"] += speed

    prev_speed = entity.get("prev_speed", 0)
    time_diff = entity.get("time_diff", 1)
    total_features["avg_acceleration"] += (speed - prev_speed) / time_diff if time_diff != 0 else 0

    location = entity.get("transform", {}).get("location", {})
    total_features["min_x_dist"] += location.get("x", 0)
    total_features["min_y_dist"] += location.get("y", 0)

    ttc = (location.get("x", 0) + location.get("y", 0)) / speed if speed != 0 else 10000
    total_features["ttc"] += ttc if not np.isinf(ttc) else 0  # Handle inf by replacing it with 0

    return True


def remove_outliers_iqr(df, columns, iqr_multiplier=3):
    """
    Remove outliers from specified columns in a DataFrame using the IQR method.

    Parameters:
    - df: DataFrame, the data to process
    - columns: list of str, the columns to apply the IQR method on
    - iqr_multiplier: float, the multiplier for the IQR to determine outlier thresholds.

    Returns:
    - DataFrame with outliers removed.
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        # Remove rows outside of bounds
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    print(f"Data size after outlier removal: {len(df)} rows.")
    return df


def train_and_plot_gmm_with_correct_order_and_scatter(df,filename_print):
    """
    Trains a GMM model on given data, projects it to 2D space using PCA, and visualizes the density.
    Prints the (x, y, p) for all points in the DataFrame.

    Parameters:
    - df: DataFrame containing the features.
    - target_json_filepath: str, path to a target JSON file to calculate its probability.
    - n_components: int, number of GMM components.
    """
    # Step 1: Remove outliers
    df = remove_outliers_iqr(df, ['avg_speed', 'avg_acceleration', 'min_x_dist', 'min_y_dist', 'ttc'])

    # Step 2: Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['avg_speed', 'avg_acceleration', 'min_x_dist', 'min_y_dist', 'ttc']].values)

    # Step 3: Fit GMM
    gmm = GaussianMixture(n_components=1, random_state=0)
    gmm.fit(X)
    with open("trained_gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)

    # Step 4: PCA transformation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Calculate probability densities for all data points
    probabilities = np.exp(gmm.score_samples(X))


    # Print (x, y, p) for all points
    print("Data points and their probabilities:")
    for i in range(len(X_pca)):
        x, y = X_pca[i]
        p = probabilities[i]
        print(f"Point {i}: (x={x:.2f}, y={y:.2f}, p={p:.5f}),filename = {df['filename'].iloc[i]}")
        if filename_print in df['filename'].iloc[i]:
            point_pca = X_pca[i]
            probability_density = p

    # Plotting the density surface
    x = np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 150)
    y = np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 150)
    X_grid, Y_grid = np.meshgrid(x, y)
    XY_grid = np.array([X_grid.ravel(), Y_grid.ravel()]).T
    XY_grid_high_dim = pca.inverse_transform(XY_grid)
    Z = np.exp(gmm.score_samples(XY_grid_high_dim)).reshape(X_grid.shape)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.contourf(X_grid, Y_grid, Z, zdir='z', offset=-4, cmap='viridis', alpha=0.9)

    mark_point_on_3d_plot(ax, point_pca[0], point_pca[1], probability_density)  #

    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0))

    ax.grid(False)
    ax.set_facecolor((1.0, 1.0, 1.0, 0))
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.w_xaxis.line.set_color('black')
    ax.w_yaxis.line.set_color('black')
    ax.w_zaxis.line.set_color('black')
    ax.set_title("3D Density Surface of GMM after PCA")
    ax.set_xlabel("PCA x")
    ax.set_ylabel("PCA y")
    ax.set_zlabel("Probability Density")

    ax.set_zlim(-4, Z.max() + 0.5)
    ax.view_init(elev=10, azim=140)
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig("correct_density_plot_3D_with_texture.png", dpi=300)
    plt.show()
    plt.close(fig)

    print("3D density plot with texture saved as correct_density_plot_3D_with_texture.png")


def mark_point_on_3d_plot(ax, x, y, z):
    """
    Mark a point on the 3D plot and annotate its z-axis value (p=z) with an angled line.

    Parameters:
    - ax: Matplotlib 3D axis object.
    - x, y,z: Coordinates of the point to be marked.
    """
    ax.scatter(x, y, z, color='r', s=50)
    ax.scatter(x, y, -3, color='r', s=50)
    ax.plot([x, x], [y, y], [-3, z], color='r', linestyle='--')

    # Draw an angled line for annotation
    x_offset = x - 2  # Slightly offset the x coordinate for the annotation line
    y_offset = y - 2  # Slightly offset the y coordinate for the annotation line
    z_offset = z + 3  # Slightly offset the z coordinate for better visibility
    ax.plot([x, x_offset], [y, y_offset], [z, z_offset], color='red', linestyle='--', linewidth=2)

    # Annotate the point with the z value
    ax.text(x_offset, y_offset, z_offset+0.5, f'p = {z:.3f}', color='black', fontsize=15, ha='center')


# Example usage
if __name__ == "__main__":
    base_folder_path = "../data"  # Path to the base folder containing subdirectories with JSON files
    filename_print = "/213/time_record/gid:2_sid:2.json"

    df = gather_data(base_folder_path)
    train_and_plot_gmm_with_correct_order_and_scatter(df,filename_print)
