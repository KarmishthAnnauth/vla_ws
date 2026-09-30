//C++ library includes
#include <memory>
#include <chrono>
#include <math.h>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <functional>
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <geometry_msgs/msg/transform_stamped.hpp>

//other macros
#define _USE_MATH_DEFINES


using std::placeholders::_1;
using namespace std::chrono_literals;


class WaypointVisualiser : public rclcpp::Node {

public:
    WaypointVisualiser() : Node("waypoint_visualiser_node")
    {
        this->declare_parameter("waypoints_path", "/sim_ws/src/pure_pursuit/racelines/e7_floor5.csv");
        this->declare_parameter("rviz_waypoints_topic", "/waypoints");
        // nav_msgs/Path copy of the same waypoints for an external planner
        // (the SimLingo node on the Orin). Latched (transient local) so a
        // subscriber that starts later, or reconnects over the network, still
        // receives it; also re-sent on the timer.
        this->declare_parameter("global_path_topic", "/global_path");

        waypoints_path = this->get_parameter("waypoints_path").as_string();
        rviz_waypoints_topic = this->get_parameter("rviz_waypoints_topic").as_string();
        global_path_topic = this->get_parameter("global_path_topic").as_string();

        vis_path_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>(rviz_waypoints_topic, 1000);
        auto latched = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
        global_path_pub = this->create_publisher<nav_msgs::msg::Path>(global_path_topic, latched);
        timer_ = this->create_wall_timer(2000ms, std::bind(&WaypointVisualiser::timer_callback, this));

        RCLCPP_INFO (this->get_logger(), "this node has been launched");
        download_waypoints();

    }

    private:
    struct csvFileData{
        std::vector<double> X;
        std::vector<double> Y;
    };


    //topic names
    std::string waypoints_path;
    std::string rviz_waypoints_topic;
    std::string global_path_topic;
    
    //file object
    std::fstream csvFile_waypoints; 

    //struct initialisation
    csvFileData waypoints;

    //Publisher initialisation
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_path_pub; 
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_pub;

    //Timer initialisation
    rclcpp::TimerBase::SharedPtr timer_;

    //private functions
    
    void download_waypoints () { //put all data in vectors
        csvFile_waypoints.open(waypoints_path, std::ios::in);

        RCLCPP_INFO (this->get_logger(), "%s", (csvFile_waypoints.is_open() ? "fileOpened" : "fileNOTopened"));
        
        //std::vector<std::string> row;
        std::string line, word, temp;

        while (!csvFile_waypoints.eof()) { //TODO: use eof to find end of file (if this doesn't work)
            
            //row.clear(); //empty row extraction string vector

            std::getline(csvFile_waypoints, line, '\n');
            //RCLCPP_INFO (this->get_logger(), "%s", line);


            std::stringstream s(line); //TODO: continue implementation

            int j = 0;
            while (getline(s, word, ',')) {

                RCLCPP_INFO (this->get_logger(), "%s", (word.empty() ? "wordempty" : "wordNOTempty"));
                //RCLCPP_INFO (this->get_logger(), "%f", std::stod(word));
                if (!word.empty()) {
                    if (j == 0) {
                        double x = std::stod(word);
                        waypoints.X.push_back(x);
                        RCLCPP_INFO (this->get_logger(), "%f... Xpoint", waypoints.X.back());
                    }
                    
                    if (j == 1) {
                        waypoints.Y.push_back(std::stod(word));
                        RCLCPP_INFO (this->get_logger(), "%f... Ypoint", waypoints.Y.back());
                    }
                }

                j++;
            }
        }

        csvFile_waypoints.close();
    }

    void visualize_points() {
        auto marker_array = visualization_msgs::msg::MarkerArray();
        auto marker = visualization_msgs::msg::Marker();
        marker.header.frame_id = "map";
        marker.header.stamp = rclcpp::Clock().now();
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.15;
        marker.scale.y = 0.15;
        marker.scale.z = 0.15;
        marker.color.a = 1.0; 
        marker.color.g = 1.0;

        for(unsigned int i=0; i<waypoints.X.size(); ++i) {
            marker.pose.position.x = waypoints.X[i];
            marker.pose.position.y = waypoints.Y[i];
            marker.id = i;
            marker_array.markers.push_back(marker);
        }

        vis_path_pub->publish(marker_array);
    }

    void publish_global_path() {
        auto path = nav_msgs::msg::Path();
        path.header.frame_id = "map";
        path.header.stamp = rclcpp::Clock().now();
        for (unsigned int i = 0; i < waypoints.X.size(); ++i) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = path.header;
            ps.pose.position.x = waypoints.X[i];
            ps.pose.position.y = waypoints.Y[i];
            // heading towards the next waypoint (last one repeats the previous)
            unsigned int j = (i + 1 < waypoints.X.size()) ? i + 1 : i;
            unsigned int k = (j == i && i > 0) ? i - 1 : i;
            double yaw = std::atan2(waypoints.Y[j] - waypoints.Y[k], waypoints.X[j] - waypoints.X[k]);
            ps.pose.orientation.z = std::sin(yaw / 2.0);
            ps.pose.orientation.w = std::cos(yaw / 2.0);
            path.poses.push_back(ps);
        }
        global_path_pub->publish(path);
    }

    void timer_callback () {
        visualize_points();
        publish_global_path();
    }
    

};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node_ptr = std::make_shared<WaypointVisualiser>(); // initialise node pointer
    rclcpp::spin(node_ptr);
    rclcpp::shutdown();
    return 0;
}
