import os
import json

import pandas as pd


import numpy as np
import matplotlib.pyplot as plt

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
    features = {
        "timestamp": [],
        "entity_id": [],
        "entity_type": [],
        "avg_speed": [],
        "avg_acceleration": [],
        "min_x_dist": [],
        "min_y_dist": [],
        "ttc": []
    }
    prev_velocity = {}

    for root, dirs, files in os.walk(base_folder_path):
        for filename in files:
            if filename.endswith('.json') and 'time_record' in root:
                filepath = os.path.join(root, filename)
                print(f"Processing file: {filepath}")

                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filepath}, skipping this file.")
                    continue

                for timestamp, value in data.items():
                    if timestamp == "min_dist":
                        continue  # Skip the "min_dist" key

                    # Process player data
                    player_data = value.get("player", {})
                    if player_data:
                        process_entity_data(player_data, "player", int(timestamp), features, prev_velocity)

                    # Process each NPC data
                    npc_data = value.get("NPC", [])
                    for npc in npc_data:
                        process_entity_data(npc, "NPC", int(timestamp), features, prev_velocity)

    # Convert to DataFrame
    df = pd.DataFrame(features)
    # Drop rows with NaN or inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def process_entity_data(entity, entity_type, timestamp, features, prev_velocity):
    """
    Processes data for an individual entity (player or NPC), extracting relevant features
    and appending them to the features dictionary.

    Parameters:
    - entity: dict, data for the entity.
    - entity_type: str, type of the entity ("player" or "NPC").
    - timestamp: int, timestamp of the current record.
    - features: dict, feature storage dictionary to append data to.
    - prev_velocity: dict, stores previous velocity values to calculate acceleration.
    """
    entity_id = entity.get("id")

    # Extract location
    location = entity.get("transform", {}).get("location", {})
    x = location.get("x", 0)
    y = location.get("y", 0)
    z = location.get("z", 0)

    # Calculate minimum x and y distances
    min_x_dist = x
    min_y_dist = y

    # Extract velocity and calculate speed
    velocity = entity.get("velocity", {})
    vx, vy, vz = velocity.get("x", 0), velocity.get("y", 0), velocity.get("z", 0)

    # Check for NaN in velocity components
    if np.isnan(vx) or np.isnan(vy) or np.isnan(vz):
        print(f"Invalid velocity data for entity {entity_id} at timestamp {timestamp}, skipping this entry.")
        return

    # Calculate speed
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    # Calculate acceleration
    prev_v = prev_velocity.get(entity_id, 0)
    time_diff = timestamp - prev_velocity.get("prev_time", timestamp)
    acceleration = (speed - prev_v) / time_diff if prev_v and time_diff != 0 else 0
    prev_velocity[entity_id] = speed
    prev_velocity["prev_time"] = timestamp

    # Calculate TTC (assuming minimum distance in both x and y)
    ttc = (x + y) / speed if speed != 0 else np.inf

    # Append data to features
    features["timestamp"].append(timestamp)
    features["entity_id"].append(entity_id)
    features["entity_type"].append(entity_type)
    features["avg_speed"].append(speed)
    features["avg_acceleration"].append(acceleration)
    features["min_x_dist"].append(min_x_dist)
    features["min_y_dist"].append(min_y_dist)
    features["ttc"].append(ttc)


def remove_outliers_iqr(df, columns):
    """
    Remove outliers from specified columns in a DataFrame using the IQR method.

    Parameters:
    - df: DataFrame, the data to process
    - columns: list of str, the columns to apply the IQR method on

    Returns:
    - DataFrame with outliers removed
    """
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df




def train_and_plot_gmm_with_correct_order_and_scatter(df, n_components=1):
    # Step 1: Remove outliers using IQR method
    df = remove_outliers_iqr(df, ['avg_speed', 'avg_acceleration', 'min_x_dist', 'min_y_dist', 'ttc'])

    # Step 2: Standardize the data
    X = df[['avg_speed', 'avg_acceleration', 'min_x_dist', 'min_y_dist', 'ttc']].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Step 3: Fit GMM on high-dimensional standardized data
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(X)

    # Step 4: Save the trained GMM model
    with open("trained_gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)
    print("GMM model saved as trained_gmm_model.pkl")

    # Step 5: Project original data to 2D space using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Step 6: Plot scatter plot of PCA-transformed data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, edgecolor='k')
    plt.title("PCA-transformed Data Scatter Plot")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig("pca_scatter_plot.png", dpi=300)
    plt.close()
    print("Scatter plot of PCA-transformed data saved as pca_scatter_plot.png")

    # Step 7: Create grid for density visualization
    x = np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 150)
    y = np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 150)
    X_grid, Y_grid = np.meshgrid(x, y)
    XY_grid = np.array([X_grid.ravel(), Y_grid.ravel()]).T

    # Step 8: Transform the 2D grid back to the original high-dimensional space for GMM scoring
    XY_grid_high_dim = pca.inverse_transform(XY_grid)

    # Step 9: Calculate GMM density on the original high-dimensional space, then reshape it for the 2D grid
    Z = np.exp(gmm.score_samples(XY_grid_high_dim)).reshape(X_grid.shape)

    # Step 10: Plotting the 2D density plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X_grid, Y_grid, Z, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Probability Density')
    ax.set_title('2D Density Visualization of GMM after PCA')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    plt.savefig("correct_density_plot_2D.png", dpi=300)
    plt.close(fig)
    print("2D density plot saved as correct_density_plot_2D.png")

    # Step 11: Plotting the 3D density plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', edgecolor='none')
    ax.set_title("3D Density Surface of GMM after PCA")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("Probability Density")

    # Optional: Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=130)

    plt.savefig("correct_density_plot_3D.png", dpi=300)
    plt.close(fig)
    print("3D density plot saved as correct_density_plot_3D.png")

# Example usage
# Assuming df is your input DataFrame
# train_and_plot_gmm_with_correct_order_and_scatter(df)



# Example usage


if __name__ == "__main__":
    base_folder_path = "../data/save"  # Path to the base folder containing subdirectories with JSON files

    # Gather data from all relevant JSON files
    df = gather_data(base_folder_path)

    # Train GMM model and save final 2D and 3D plots
    train_and_plot_gmm_with_correct_order_and_scatter(df)
