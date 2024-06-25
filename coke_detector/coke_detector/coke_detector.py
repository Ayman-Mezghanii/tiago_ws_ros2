import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import onnxruntime as ort
import numpy as np
import os
import time

class CokeDetector(Node):
    def __init__(self):
        super().__init__('coke_detector')
        self.subscription = self.create_subscription(
            Image,
            '/head_front_camera/rgb/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.model_path = '/home/ayman/runs/detect/train23/weights/best.onnx'
        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.get_logger().info('Coke Detector Node has been started')
        os.makedirs('detected_images', exist_ok=True)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        img = cv2.resize(cv_image, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # Change data layout from HWC to CHW
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        ort_inputs = {self.input_name: img}
        ort_outs = self.session.run([self.output_name], ort_inputs)
        detections = ort_outs[0][0]

        self.get_logger().info(f'Detections shape: {detections.shape}')

        h, w, _ = cv_image.shape
        for detection in detections:
            x_center, y_center, width, height, conf = detection[:5]
            if conf > 0.2:  # Lowered confidence threshold
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((x_center + height / 2) * h)

                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f'Coke Can: {conf:.2f}'
                cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                self.get_logger().info(f'Coke can detected with confidence {conf:.2f}')

        # Save the image
        filename = f'detected_images/{int(time.time())}.jpg'
        cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, cv_image_bgr)
        self.get_logger().info(f'Image saved as {filename}')

def main(args=None):
    rclpy.init(args=args)
    coke_detector = CokeDetector()
    rclpy.spin(coke_detector)
    coke_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

