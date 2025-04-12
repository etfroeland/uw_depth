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
#from depth_estimation.utils.visualization import gray_to_heatmap
from depth_estimation.utils.depth_prior import get_depth_prior_from_features

# Load model once and keep in memory
MODEL_PATH = os.path.join(parent_dir, "data", "saved_models", "model_e11_udfnet_lr0.0001_bs6_lrd0.9_with_infguidance.pth")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UWDepthNode:
    def __init__(self):
        rospy.init_node("uw_depth_node", anonymous=True)

        self.use_priors = rospy.get_param("~use_priors", True)

        # OpenCV Bridge
        self.bridge = CvBridge()

        # Subscribers (Using message_filters for synchronization)
        self.image_sub = Subscriber("/ted/image", Image)
        self.prior_sub = Subscriber("/fft/depth_pointcloud", PointCloud)

        # Synchronize messages
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.prior_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        # Publisher
        self.depth_pub = rospy.Publisher("/uw_depth/depth_image", Image, queue_size=1)
        self.heatmap_pub = rospy.Publisher("/uw_depth/heatmap", Image, queue_size=1)
        self.combined_pub = rospy.Publisher("/uw_depth/combined_image", Image, queue_size=1)

        # Load Model
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

            tensor_conversion_time = time.time()

            # Convert prior to tensor
            features = torch.tensor(priors, dtype=torch.float32).unsqueeze(0)
            prior_tensor = get_depth_prior_from_features(features).to(DEVICE)            # get_depth_prior_from_features is the bottleneck
            
            # Convert image to tensor
            resized_cv_rgb = cv2.resize(cv_rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
            rgb_tensor = ToTensor()(resized_cv_rgb).unsqueeze(0).to(DEVICE)
            print("Tensor conversion time: ", time.time() - tensor_conversion_time)

            # Run inference
            start_time = time.time()
            with torch.no_grad():
                prediction, _ = self.model(rgb_tensor, prior_tensor)
            end_time = time.time()
            rospy.loginfo(f"Inference Time: {(end_time - start_time):.4f} sec")
        
            # Depth map as a 2D numpy array in meters
            depth_map = prediction[0].squeeze(0).cpu().numpy()
            depth_msg = self.bridge.cv2_to_imgmsg((depth_map).astype(np.float32), encoding="32FC1")  # Use 32-bit float encoding for depth images
            print(depth_map.mean())
            depth_msg.header = image_msg.header  # Set the header to match the input image
            depth_msg.header.frame_id = "camera_frame"
            self.depth_pub.publish(depth_msg)

            # For visualization ------------------------------------------------------------------------------------------------------------------------------
            depth_map_scaled = np.minimum(depth_map, 3.0) / 3.0  # Scale to [0, 1], values above 3m are saturated
            depth_gray = (depth_map_scaled * 255).astype(np.uint8)  # Convert to 8-bit grayscale

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
            # End of visualization ---------------------------------------------------------------------------------------------------------------------------
            print("Total_time: ", time.time() - total_time)

            # Save the priors as a csv file with filename as image seq
            # seq = image_msg.header.seq
            # prior_file = os.path.join(parent_dir, "depth_priors", f"{seq:04d}_prior.csv")
            # np.savetxt(prior_file, priors, delimiter=",", header="x,y,z", comments='')

            

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == "__main__":
    try:
        UWDepthNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
