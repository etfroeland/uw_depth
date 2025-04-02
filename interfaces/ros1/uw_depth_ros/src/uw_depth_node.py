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

# Get the parent directory four levels up
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Add the parent directory to sys.path so we can access the modules from there
sys.path.insert(0, parent_dir)

from depth_estimation.model.model import UDFNet
#from depth_estimation.utils.visualization import gray_to_heatmap
from depth_estimation.utils.depth_prior import get_depth_prior_from_features

# Load model once and keep in memory
MODEL_PATH = os.path.join(parent_dir, "data", "saved_models", "model_e11_udfnet_lr0.0001_bs6_lrd0.9_with_infguidance.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UWDepthNode:
    def __init__(self):
        rospy.init_node("uw_depth_node", anonymous=True)

        # OpenCV Bridge
        self.bridge = CvBridge()

        # # Subscribers (Using message_filters for synchronization)
        # self.image_sub = Subscriber("/camera/image_raw", Image)
        # self.prior_sub = Subscriber("/depth_estimation/prior", PointCloud)

        # # Synchronize messages
        # self.sync = ApproximateTimeSynchronizer([self.image_sub, self.prior_sub], queue_size=10, slop=0.1)
        # self.sync.registerCallback(self.callback)

        #For testing
        self.prior_sub = Subscriber("/depth_estimation/prior", PointCloud)
        # Synchronize messages
        self.sync = ApproximateTimeSynchronizer([self.prior_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)



        # Publisher
        self.depth_pub = rospy.Publisher("/uw_depth/depth_image", Image, queue_size=1)

        # Load Model
        rospy.loginfo(f"Loading model on {DEVICE}")
        self.model = UDFNet(n_bins=80).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        rospy.loginfo("Model loaded successfully.")

    def callback(self, prior_msg):
        try:
            # # Convert ROS image to OpenCV format
            # cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")

            #For testing
            image_path = os.path.join(parent_dir, "interfaces", "ros1", "uw_depth_ros", "src", "image.jpg")
            cv_rgb = cv2.imread(image_path)

            # Convert ROS pointcloud to numpy array
            points = prior_msg.points
            priors = np.array([[point.x, point.y, point.z] for point in points])
            # Convert to tensor
            features = torch.tensor(priors, dtype=torch.float32).unsqueeze(0)

            # Convert to tensor and resize
            orig_rgb_tensor = ToTensor()(cv_rgb).unsqueeze(0).to(DEVICE)
            resized_rgb_tensor = torch.nn.functional.interpolate(orig_rgb_tensor, (480, 640), mode="bilinear", align_corners=False)
            prior_tensor = get_depth_prior_from_features(features).to(DEVICE)

            # Run inference
            start_time = time.time()
            with torch.no_grad():
                prediction, _ = self.model(resized_rgb_tensor, prior_tensor)
            end_time = time.time()
            print("Prediction shape: ", prediction.shape)

            rospy.loginfo(f"Inference Time: {(end_time - start_time):.4f} sec")

            # Remove batch dimension
            heatmap = prediction[0].squeeze(0).cpu().numpy()

            # Convert the heatmap to a color image (hot color map)
            colored_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_HOT)

            # Convert to a ROS image message (appropriate encoding for color images)
            depth_msg = self.bridge.cv2_to_imgmsg(colored_heatmap, encoding="bgr8")

            # Publish result
            self.depth_pub.publish(depth_msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == "__main__":
    try:
        UWDepthNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
