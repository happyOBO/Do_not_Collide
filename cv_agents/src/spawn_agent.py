#!/usr/bin/python
#-*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import tf

import rospkg
import sys

from scipy.interpolate import interp1d

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion
from object_msgs.msg import Object

import pickle
import argparse
import optimal_trajectory_Frenet


dt = 0.1

k = 0.5  # control gain
L = 2.875

rospack = rospkg.RosPack()
path = rospack.get_path("map_server")

rn_id = dict()

rn_id[5] = {
    'left': [18, 2, 11, 6, 13, 8, 15, 10, 26, 0]  # ego route
}

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def interpolate_waypoints(wx, wy, space=0.5):
    _s = 0
    s = [0]
    for i in range(1, len(wx)):
        prev_x = wx[i - 1]
        prev_y = wy[i - 1]
        x = wx[i]
        y = wy[i]

        dx = x - prev_x
        dy = y - prev_y

        _s = np.hypot(dx, dy)
        s.append(s[-1] + _s)

    fx = interp1d(s, wx)
    fy = interp1d(s, wy)
    ss = np.linspace(0, s[-1], num=int(s[-1] / space) + 1, endpoint=True)

    dxds = np.gradient(fx(ss), ss, edge_order=1)
    dyds = np.gradient(fy(ss), ss, edge_order=1)
    wyaw = np.arctan2(dyds, dxds)

    return {
        "x": fx(ss),
        "y": fy(ss),
        "yaw": wyaw,
        "s": ss
    }


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, dt=0.1, WB=2.6):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        self.dt = dt
        self.WB = WB

    def update(self, a, delta):
        dt = self.dt
        WB = self.WB

        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.yaw = pi_2_pi(self.yaw)
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


def stanley_control(x, y, yaw, v, map_xs, map_ys, map_yaws):
    # find nearest point
    min_dist = 1e9
    min_index = 0
    n_points = len(map_xs)

    front_x = x + L * np.cos(yaw)
    front_y = y + L * np.sin(yaw)

    for i in range(n_points):
        dx = front_x - map_xs[i]
        dy = front_y - map_ys[i]

        dist = np.sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            min_dist = dist
            min_index = i

    # compute cte at front axle
    map_x = map_xs[min_index]
    map_y = map_ys[min_index]
    map_yaw = map_yaws[min_index]
    dx = map_x - front_x
    dy = map_y - front_y

    perp_vec = [np.cos(yaw + np.pi/2), np.sin(yaw + np.pi/2)]
    cte = np.dot([dx, dy], perp_vec)

    # control law
    yaw_term = normalize_angle(map_yaw - yaw)
    cte_term = np.arctan2(k*cte, v)

    # steering

    steer = yaw_term + cte_term
    return steer


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def get_ros_msg(x, y, yaw, v, id):
    quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

    m = Marker()
    m.header.frame_id = "/map"
    m.header.stamp = rospy.Time.now()
    m.id = id
    m.type = m.CUBE

    m.pose.position.x = x + 1.3 * math.cos(yaw)
    m.pose.position.y = y + 1.3 * math.sin(yaw)
    m.pose.position.z = 0.75
    m.pose.orientation = Quaternion(*quat)

    m.scale.x = 4.475
    m.scale.y = 1.850
    m.scale.z = 1.645

    m.color.r = 93 / 255.0
    m.color.g = 122 / 255.0
    m.color.b = 177 / 255.0
    m.color.a = 0.97

    o = Object()
    o.header.frame_id = "/map"
    o.header.stamp = rospy.Time.now()
    o.id = id
    o.classification = o.CLASSIFICATION_CAR
    o.x = x
    o.y = y
    o.yaw = yaw
    o.v = v
    o.L = m.scale.x
    o.W = m.scale.y

    return {
        "object_msg": o,
        "marker_msg": m,
        "quaternion": quat
    }

obs = [None, None]

def obs2_callback(data):
    global obs
    obs[0] = data

def obs3_callback(data):
    global obs
    obs[1] = data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spawn a CV agent')

    parser.add_argument("--id", "-i", type=int, help="agent id", default=1)
    parser.add_argument("--route", "-r", type=int,
                        help="start index in road network. select in [1, 3, 5, 10]", default=5)
    parser.add_argument("--dir", "-d", type=str, default="left", help="direction to go: [left, straight, right]")
    args = parser.parse_args()

    rospy.init_node("three_cv_agents_node_" + str(args.id))

    id = args.id
    tf_broadcaster = tf.TransformBroadcaster()
    marker_pub = rospy.Publisher("/objects/marker/car_" + str(id), Marker, queue_size=1)
    object_pub = rospy.Publisher("/objects/car_" + str(id), Object, queue_size=1)

    rospy.Subscriber("/objects/car_2", Object, obs2_callback, queue_size= 1)
    rospy.Subscriber("/objects/car_3", Object, obs3_callback, queue_size= 1)

    while(obs[0] == None or obs[1] == None):
        continue
    start_node_id = args.route
    route_id_list = [start_node_id] + rn_id[start_node_id][args.dir]

    ind = 100

    kp = 0.5
    kd = 0.1
    ki = 0.0

    with open(path + "/src/route.pkl", "rb") as f:
        nodes = pickle.load(f)

    wx = []
    wy = []
    wyaw = []
    for _id in route_id_list:
        wx.append(nodes[_id]["x"][1:])
        wy.append(nodes[_id]["y"][1:])
        wyaw.append(nodes[_id]["yaw"][1:])
    wx = np.concatenate(wx)
    wy = np.concatenate(wy)
    wyaw = np.concatenate(wyaw)

    waypoints = interpolate_waypoints(wx, wy)
    

    target_speed = 20.0 / 3.6
    state = State(x=waypoints["x"][ind], y=waypoints["y"][ind], yaw=waypoints["yaw"][ind], v=1.0, dt=0.01)

    int_error = 0.0
    prev_error = 0.0

    r = rospy.Rate(100)
    
    ## plannining

    mapx = waypoints["x"]
    mapy = waypoints["y"]
    maps = waypoints["s"]

    # get maps
    s, d = optimal_trajectory_Frenet.get_frenet(state.x, state.y, mapx, mapy)
    x, y, road_yaw = optimal_trajectory_Frenet.get_cartesian(s, d, mapx, mapy, maps)
    yawi = state.yaw - road_yaw

    v = 1.0
    a = 0


    # s 방향 초기조건
    si = s
    si_d = v*np.cos(yawi)
    si_dd = a*np.cos(yawi)
    sf_d = target_speed
    sf_dd = 0

    # d 방향 초기조건
    di = d
    di_d = v*np.sin(yawi)
    di_dd = a*np.sin(yawi)
    df_d = 0
    df_dd = 0

    opt_d = di

    while not rospy.is_shutdown():
        # get maps (planning)
        path, opt_ind ,succeed = optimal_trajectory_Frenet.frenet_optimal_planning(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, mapx, mapy, maps, opt_d)

        if(succeed):
            print("success")
            di = stanley_control(state.x, state.y, state.yaw, state.v, path[opt_ind].x, path[opt_ind].y ,path[opt_ind].yaw)
            si = path[0].s[1]
            si_d = path[0].s_d[1]
            si_dd = path[0].s_dd[1]
            di_d = path[0].d_d[1]
            di_dd = path[0].d_dd[1]
        
        else :
            s,d = optimal_trajectory_Frenet.get_frenet(state.x, state.y, mapx, mapy)
            x,y,road_yaw = optimal_trajectory_Frenet.get_cartesian(s, d, mapx, mapy, maps)
            steer = road_yaw - state.yaw
            di = stanley_control(state.x, state.y, state.yaw, state.v, waypoints["x"], waypoints["y"] ,waypoints["yaw"])
        
        opt_d = di

        speed_error = target_speed - state.v
        diff_error = speed_error - prev_error
        prev_error = speed_error
        int_error += speed_error

        # di ,prev_error , int_error = pid_control(state.x, state.y, state.yaw, state.v, waypoints["x"], waypoints["y"], waypoints["yaw"], int_error,prev_error)
        ai = kp * speed_error - kd * diff_error/dt - ki * int_error

        # update state with acc, delta
        state.update(ai, di)

        # vehicle state --> topic msg
        msg = get_ros_msg(state.x, state.y, state.yaw, state.v, id=id)

        # send tf
        tf_broadcaster.sendTransform(
            (state.x, state.y, 1.5),
            msg["quaternion"],
            rospy.Time.now(),
            "/car_" + str(id), "/map"
        )

        # publish vehicle state in ros msg
        object_pub.publish(msg["object_msg"])
        ind+= 1

        r.sleep()
