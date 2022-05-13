# Authors: Howen Anthony Fernando, Jessica Vogt, Cody Crawford
# RedIDs: 822914112, 823660429, 824167663
# Class: CS549, Spring 2022
# Assignment: Pose Gesture Analysis

import proj
from proj import np
import sys
import cv2
import mediapipe as mp
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import pandas as pd


"""
Extracts pose data using the mediapipe pose library.
Then uses the pose_embedder model to convert the pose data
into embedding vectors.
"""


def extract_and_embed_poses(pose_embedder, filename, embeddings):
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(filename)

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # check that actually captured landmarks!
            try:
                landmarks = np.array(results.pose_landmarks.landmark)
            except:
                continue

            embedding = pose_embedder.__call__(landmarks=landmarks, vidname=filename)

            embeddings.append(embedding)

    cap.release()


"""
Takes in the embeddings dataframe and a boolean for the cluster mode.
Uses plotly to visualize the data.
"""


def visualize_data(df_emb, cluster_mode):
    features = df_emb.drop("videoname", axis=1)

    if cluster_mode:
        features = features.drop("cluster", axis=1)

    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(features)

    if cluster_mode:
        vidName = df_emb["videoname"].iloc[0]
        fig = px.scatter(
            projections,
            color=df_emb.cluster,
            x=0,
            y=1,
            title=f"Gaussian Clustering on {vidName}",
        )
    else:
        fig = px.scatter(
            projections,
            color=df_emb.videoname,
            x=0,
            y=1,
            title="Pose Embeddings of multiple videos overlaid",
        )

    fig.show()


"""
Creates clusters for the dataframe using Gaussian Mixture
clustering. Adds the cluster info as a column to the dataframe.
Returns the altered dataframe.
"""


def gaussian_mix(df_emb):
    gaussMix = GaussianMixture(n_components=3)
    df_emb_numeric = df_emb.drop("videoname", axis=1)
    gaussMix.fit(df_emb_numeric)
    cluster = gaussMix.predict(df_emb_numeric)
    df_emb["cluster"] = cluster
    return df_emb


"""
Convert embedding numpy array into a dataframe.
Gives the dataframe column names based on an index
and the last column name as videoname.
Returns the dataframe.
"""


def embeddings2Df(embeddings):
    return pd.DataFrame(
        embeddings,
        columns=[
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "videoname",
        ],
    )


def main():
    cluster_mode = True  # cluster_mode is True if only 1 video

    # set cluster mode false if more than 1 vid supplied
    if len(sys.argv) > 2:
        cluster_mode = False

    pose_embedder = proj.FullBodyPoseEmbedder()
    embedding_list = []

    # iterate through list of videos, extract and embed
    for arg in sys.argv[1:]:
        extract_and_embed_poses(pose_embedder, arg, embedding_list)

    embedding_arr = np.array(embedding_list)  # convert list to np array
    df_emb = embeddings2Df(embedding_arr)  # convert np array to pandas df

    if cluster_mode:
        df_emb = gaussian_mix(df_emb)

    visualize_data(df_emb, cluster_mode)


if __name__ == "__main__":
    main()
