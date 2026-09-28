import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

class Safety(Node):
    def __init__(self):
        # Initialize this node with the name 'safety_node'
        super().__init__('safety_node')
        
        # Initialize the car's velocity to 0.0
        self.v_car = 0.0
        
        # Task 2: Initialize the emergency brake status. '0' means brake is not activated.
        self.emergency_brake = 0

        # Subscribe to the LaserScan topic "/scan". When a message is received, 'scan_callback' will be executed.
        self.laser_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        
        # Task 1: Subscribe to the Odometry topic "/odom". When a message is received, 'odom_callback' will be executed.
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        
        # Publisher to publish messages of type AckermannDriveStamped to the topic "/drive".
        self.brake_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

    # Callback function for the odometry subscription
    def odom_callback(self, odom_msg):
        # Extract the linear velocity of the car from the odometry message and store it
        self.v_car = odom_msg.twist.twist.linear.x

    # Callback function for the laser scan subscription
    def scan_callback(self, scan_msg):
        # Convert the list of range measurements from the LaserScan message to a numpy array
        ranges = np.array(scan_msg.ranges, dtype=float)

        #self.get_logger().info("ranges len():" + str(len(ranges))) # 1081
        
        # The power connector for the Jetson in picked up by the lidar, so I overwrite the values to inf to make the code work
        ranges[1000:] = np.inf

        # Find the minimum value and its index
        min_value = np.min(ranges)
        min_index = np.argmin(ranges)
               

    
        # Calculate the angles corresponding to each range measurement using the angular properties from the LaserScan message
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

        # Define constants for braking
        BRAKING_ACCELERATION = 1.0  # Maximum deceleration rate (m/s^2) # value got experimentaly
        
        # Calculate the stopping distance (d = v^2 / (2 * a)) and stopping time (t = v / a)
        stopping_distance = self.v_car**2 / (2 * BRAKING_ACCELERATION)
        stopping_time = self.v_car / BRAKING_ACCELERATION

        ## Calculate the threshold TTC based on stopping time
        #ttc_threshold = stopping_time
        
        # Define a threshold for Time To Collision (TTC)
        ttc_threshold = max(0.5, stopping_time)
        
        # Calculate TTC for each valid range reading
        TTC = [rng / max(self.v_car * np.cos(angle), 0.0001) for rng, angle in zip(ranges, angles) if not np.isnan(rng) and rng > 0.0]

        # Determine the minimum TTC from the list
        min_ttc = min(TTC, default=float('inf'))

        # Determine if an emergency brake should be applied based on the minimum TTC
        should_brake = min_ttc < ttc_threshold
        
        # Create a new AckermannDriveStamped message
        ackermann_msg = AckermannDriveStamped()


        # Log the length, minimum value, and index of the minimum value
        self.get_logger().info(f"min_dist: {min_value}, index: {min_index}, Speed: {self.v_car}, Brake: {should_brake}") 


        ### Task 2: Decide the action based on whether braking is necessary or not ###
    
        # Logic for braking and driving
        if self.v_car > 0.01:  # Car is moving
            if should_brake:
                # Start or continue braking
                #self.get_logger().info("Braking...")
                ackermann_msg.drive.speed = 0.0
            else:
                # Drive normally
                ackermann_msg.drive.speed = 1.0
        else:  # Car is effectively stopped
            if min_value < 0.5:
                # Keep braking if obstacle is too close
                #self.get_logger().info("Obstacle too close, holding brake...")
                ackermann_msg.drive.speed = 0.0
            else:
                # Resume driving if safe
                #self.get_logger().info("Resuming driving...")
                ackermann_msg.drive.speed = 1.0

        # Publish the AckermannDriveStamped message, which either commands the car to stop or move at 2 m/s
        self.brake_pub.publish(ackermann_msg)

        # If the minimum TTC is less than ttc_threshold, log the value for debugging purposes
        if min_ttc < ttc_threshold:
            self.get_logger().info(f"Minimum TTC: {min_ttc}")


def main(args=None):
    rclpy.init(args=args)
    node = Safety()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

