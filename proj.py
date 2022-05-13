# Authors: Howen Anthony Fernando, Jessica Vogt, Cody Crawford
# RedIDs: 822914112
# Class: CS549, Spring 2022
# Assignment: Pose Gesture Analysis

from math import sqrt
import numpy as np


class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky_1",
            "right_pinky_1",
            "left_index_1",
            "right_index_1",
            "left_thumb_2",
            "right_thumb_2",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

    def __call__(self, landmarks, vidname):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(
            self._landmark_names
        ), "Unexpected number of landmarks: {}".format(landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks, vidname)

        return embedding

    def _get_pose_distance_embedding(self, landmarks, vidname):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array(
            [
                # One joint.
                self._get_distance(
                    self._get_average_by_names(landmarks, "left_hip", "right_hip"),
                    self._get_average_by_names(
                        landmarks, "left_shoulder", "right_shoulder"
                    ),
                ),
                self._get_distance_by_names(landmarks, "left_shoulder", "left_elbow"),
                self._get_distance_by_names(landmarks, "right_shoulder", "right_elbow"),
                self._get_distance_by_names(landmarks, "left_elbow", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_elbow", "right_wrist"),
                self._get_distance_by_names(landmarks, "left_hip", "left_knee"),
                self._get_distance_by_names(landmarks, "right_hip", "right_knee"),
                self._get_distance_by_names(landmarks, "left_knee", "left_ankle"),
                self._get_distance_by_names(landmarks, "right_knee", "right_ankle"),
                # Two joints.
                self._get_distance_by_names(landmarks, "left_shoulder", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_shoulder", "right_wrist"),
                self._get_distance_by_names(landmarks, "left_hip", "left_ankle"),
                self._get_distance_by_names(landmarks, "right_hip", "right_ankle"),
                # Four joints.
                self._get_distance_by_names(landmarks, "left_hip", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_hip", "right_wrist"),
                # Five joints.
                self._get_distance_by_names(landmarks, "left_shoulder", "left_ankle"),
                self._get_distance_by_names(landmarks, "right_shoulder", "right_ankle"),
                self._get_distance_by_names(landmarks, "left_hip", "left_wrist"),
                self._get_distance_by_names(landmarks, "right_hip", "right_wrist"),
                # Cross body.
                self._get_distance_by_names(landmarks, "left_elbow", "right_elbow"),
                self._get_distance_by_names(landmarks, "left_knee", "right_knee"),
                self._get_distance_by_names(landmarks, "left_wrist", "right_wrist"),
                self._get_distance_by_names(landmarks, "left_ankle", "right_ankle"),
                vidname,
            ]
        )

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = np.empty(3)
        lmk_from[0] = landmarks[self._landmark_names.index(name_from)].x
        lmk_from[1] = landmarks[self._landmark_names.index(name_from)].y
        lmk_from[2] = landmarks[self._landmark_names.index(name_from)].z

        lmk_to = np.empty(3)
        lmk_to[0] = landmarks[self._landmark_names.index(name_to)].x
        lmk_to[1] = landmarks[self._landmark_names.index(name_to)].y
        lmk_to[2] = landmarks[self._landmark_names.index(name_to)].z

        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = np.empty(3)
        lmk_from[0] = landmarks[self._landmark_names.index(name_from)].x
        lmk_from[1] = landmarks[self._landmark_names.index(name_from)].y
        lmk_from[2] = landmarks[self._landmark_names.index(name_from)].z

        lmk_to = np.empty(3)
        lmk_to[0] = landmarks[self._landmark_names.index(name_to)].x
        lmk_to[1] = landmarks[self._landmark_names.index(name_to)].y
        lmk_to[2] = landmarks[self._landmark_names.index(name_to)].z

        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        dist = sqrt(
            (lmk_from[0] - lmk_to[0]) ** 2
            + (lmk_from[1] - lmk_to[1]) ** 2
            + (lmk_from[2] - lmk_to[2]) ** 2
        )
        return dist
