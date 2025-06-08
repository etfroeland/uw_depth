#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud
from message_filters import ApproximateTimeSynchronizer, Subscriber

import torch
import time
import cv2
import numpy as np
from cv_bridge import CvBridge
from torchvision.transforms import ToTensor
import sys
import os
from PIL import Image as PILImage

# Get the parent directory four levels up
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Add the parent directory to sys.path so we can access the modules from there
sys.path.insert(0, parent_dir)

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.depth_prior import get_depth_prior_from_features

# Load model once and keep in memory
MODEL_PATH = os.path.join(parent_dir, "data", "saved_models", "model_e11_udfnet_lr0.0001_bs6_lrd0.9_with_infguidance.pth")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UWDepthNode:
    def __init__(self):
        rospy.init_node("uw_depth_node", anonymous=True)

        # Initialize parameters
        self.sub_image_topic = rospy.get_param("~sub_image_topic", "/camera/image_topic")
        self.sub_prior_topic = rospy.get_param("~sub_prior_topic", "/prior_topic")
        self.depth_topic = rospy.get_param("~depth_topic", "/uw_depth/depth_image")
        self.heatmap_topic = rospy.get_param("~heatmap_topic", "/uw_depth/heatmap")
        self.combined_topic = rospy.get_param("~combined_topic", "/uw_depth/combined_image")
        self.use_priors = rospy.get_param("~use_priors", True)
        self.max_priors = rospy.get_param("~max_priors", 300)

        # Bridge and publishers
        self.bridge = CvBridge()
        self.depth_pub = rospy.Publisher(self.depth_topic, Image, queue_size=1)
        self.heatmap_pub = rospy.Publisher(self.heatmap_topic, Image, queue_size=1)
        self.combined_pub = rospy.Publisher(self.combined_topic, Image, queue_size=1)

        # Synchronized subscribers
        self.image_sub = Subscriber(self.sub_image_topic, Image)
        self.prior_sub = Subscriber(self.sub_prior_topic, PointCloud)
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.prior_sub], queue_size=30, slop=0.05)
        self.sync.registerCallback(self.callback)

        # Load depth estimation model
        rospy.loginfo(f"Loading model on {DEVICE}")
        self.model = UDFNet(n_bins=80).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        rospy.loginfo("Model loaded successfully.")


    def callback(self, image_msg, prior_msg):
        rospy.loginfo("Received image and point cloud messages.")
        
        try:
            total_time = time.time()

            # Convert ROS image to OpenCV format
            cv_rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

            # Convert ROS pointcloud to numpy array
            points = prior_msg.points
            priors = np.array([[point.x, point.y, point.z] for point in points], dtype=np.float32)
            rospy.loginfo(f"Number of prior points: {len(priors)}")

            # Pick a random subset of priors
            if self.use_priors:
                max_priors = self.max_priors
                if len(priors) > max_priors:
                    indices = np.random.choice(len(priors), max_priors, replace=False)
                    priors = priors[indices]
                elif len(priors) < max_priors:
                    # If there are fewer points than max_priors, repeat the points
                    priors = np.tile(priors, (max_priors // len(priors) + 1, 1))[:max_priors]

            # Convert prior to tensor
            features = torch.tensor(priors, dtype=torch.float32).unsqueeze(0)
            prior_tensor = get_depth_prior_from_features(features).to(DEVICE) 
            
            # Convert image to tensor
            resized_cv_rgb = cv2.resize(cv_rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
            rgb_tensor = ToTensor()(resized_cv_rgb).unsqueeze(0).to(DEVICE)

            # Run inference
            start_time = time.time()
            with torch.no_grad():
                prediction, _ = self.model(rgb_tensor, prior_tensor)
            end_time = time.time()
            rospy.loginfo(f"Inference Time: {(end_time - start_time):.4f} sec")
        
            # Depth map as a 2D numpy array in meters
            depth_map = prediction[0].squeeze(0).cpu().numpy()
            depth_msg = self.bridge.cv2_to_imgmsg((depth_map).astype(np.float32), encoding="32FC1")
            depth_msg.header = image_msg.header  # Set the header to match the input image
            self.depth_pub.publish(depth_msg)

            # For visualization ------------------------------------------------------------------------------------------------------------------------------
            depth_gray = (depth_map * 255).astype(np.uint8)  # Convert to 8-bit grayscale

            # Apply inferno colormap for consistent visualization
            colored_heatmap = cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)
            heatmap_msg = self.bridge.cv2_to_imgmsg(colored_heatmap, encoding="bgr8")

            # Resize RGB to match heatmap size
            resized_cv_rgb = cv2.resize(resized_cv_rgb, (colored_heatmap.shape[1], colored_heatmap.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Combine RGB and heatmap horizontally
            combined_image = np.hstack((resized_cv_rgb, colored_heatmap))
            combined_image_msg = self.bridge.cv2_to_imgmsg(combined_image, encoding="bgr8")

            # Publish vizualisation result
            self.heatmap_pub.publish(heatmap_msg)
            self.combined_pub.publish(combined_image_msg)
            

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == "__main__":
    try:
        UWDepthNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
