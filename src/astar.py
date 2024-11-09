#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import tf_transformations
import yaml
import os
from PIL import Image
from rclpy.qos import QoSProfile, DurabilityPolicy

# AStarPlanner 임포트
from utils import AStarPlanner

class AStarNavigationNode(Node):
    def __init__(self):
        super().__init__('astar_navigation')

        # 파라미터 선언: 이미 선언되었는지 확인 후 선언
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
            self.get_logger().info("Declared parameter 'use_sim_time'")
        else:
            self.get_logger().info("Parameter 'use_sim_time' already declared")

        use_sim_time = self.get_parameter('use_sim_time').value
        self.get_logger().info(f"use_sim_time: {use_sim_time}")

        # 맵 파일 경로 설정
        self.map_yaml_file = '/home/teus/tb3_ws/src/astar/maps/carter_warehouse_navigation.yaml'
        self.map_png_file = '/home/teus/tb3_ws/src/astar/maps/carter_warehouse_navigation.pgm'

        # 맵 로드
        self.occupancy_grid, self.map_info = self.load_map(self.map_yaml_file)

        # A* 플래너 초기화 (맵의 origin 전달)
        self.robot_radius = 0.1  # 로봇의 반경 설정
        self.astar_planner = AStarPlanner(
            self.occupancy_grid,  # occupancy_grid를 직접 전달
            self.map_info['resolution'],
            self.robot_radius,
            self.map_info['origin']  # 맵의 origin 전달
        )
        self.get_logger().info('A* planner initialized')

        # Odometry 구독자 및 cmd_vel 퍼블리셔 설정
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        self.current_pose = None  # PoseStamped 대신 Pose 사용

        # 목표 위치 설정 (예: x=2.0, y=0.0)
        self.goal_pose = PoseStamped()
        self.goal_pose.header.frame_id = 'odom'
        self.goal_pose.pose.position.x = -6.0
        self.goal_pose.pose.position.y = 8.0  # y 좌표를 0으로 설정 (필요에 따라 변경)

        # 경로 저장용 변수
        self.path_x = []
        self.path_y = []
        self.path_index = 0
        self.path_planned = False  # 경로가 이미 계획되었는지 확인하는 플래그

        # 타이머를 사용하여 주기적으로 control_loop를 호출
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # 경로 시각화를 위한 퍼블리셔 추가 (QoS 설정 변경)
        qos_profile = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.path_publisher = self.create_publisher(Path, '/planned_path', qos_profile)
        self.get_logger().info('Path publisher initialized with TRANSIENT_LOCAL durability')

        # 맵 퍼블리셔 추가 (QoS 설정 변경)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', qos_profile)
        self.publish_map()

    def load_map(self, map_yaml_file):
        # 맵 YAML 파일 로드
        with open(map_yaml_file, 'r') as file:
            map_data = yaml.safe_load(file)

        # 맵 이미지 로드
        map_image_path = os.path.join(os.path.dirname(map_yaml_file), map_data['image'])
        self.get_logger().info(f'Map image path: {map_image_path}')
        map_image = Image.open(map_image_path).convert('L')
        map_array = np.array(map_image)

        # 점유 그리드 생성 (0: 비어 있음, 100: 장애물, -1: 알 수 없음)
        # 현재는 알 수 없는 영역을 고려하지 않고 0과 100으로 설정
        occupancy_grid = np.where(map_array < 128, 100, 0).astype(np.int8)
        self.get_logger().info(f"Occupancy grid unique values: {np.unique(occupancy_grid)}")

        map_info = {
            'resolution': map_data['resolution'],
            'origin': map_data['origin'],  # 원점을 YAML 파일에서 읽어옴
            'width': occupancy_grid.shape[1],
            'height': occupancy_grid.shape[0],
        }

        return occupancy_grid, map_info

    def publish_map(self):
        # OccupancyGrid 메시지 생성 및 발행
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = 'map'
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.info.resolution = self.map_info['resolution']
        map_msg.info.width = self.map_info['width']
        map_msg.info.height = self.map_info['height']
        map_msg.info.origin.position.x = self.map_info['origin'][0]
        map_msg.info.origin.position.y = self.map_info['origin'][1]
        map_msg.info.origin.orientation.w = 1.0  # 방향은 중요하지 않으므로 단위 쿼터니언 사용
        # OccupancyGrid 데이터 설정
        flat_occupancy_grid = self.occupancy_grid.flatten(order='C')
        map_msg.data = flat_occupancy_grid.tolist()
        self.map_publisher.publish(map_msg)
        self.get_logger().info('Published map to /map topic.')

    def odom_callback(self, msg):
        # Odometry 메시지의 프레임 확인
        self.get_logger().debug(f"Odometry message frame_id: {msg.header.frame_id}")
        # 로봇의 현재 위치를 업데이트
        self.current_pose = msg.pose.pose  # PoseStamped가 아니라 Pose로 저장

    def plan_path(self):
        if self.astar_planner is None:
            return False

        if self.current_pose is None:
            self.get_logger().info('Odometry 데이터 기다리는 중...')
            return False

        # 현재 위치는 이미 odom 프레임에 있으므로, TF 변환이 필요 없음
        sx = self.current_pose.position.x
        sy = self.current_pose.position.y
        gx = self.goal_pose.pose.position.x
        gy = self.goal_pose.pose.position.y

        # 시작점과 목표점이 맵 내에 있는지 확인
        if not self.astar_planner.verify_position(sx, sy):
            self.get_logger().error('시작 위치가 유효하지 않거나 장애물입니다.')
            return False
        if not self.astar_planner.verify_position(gx, gy):
            self.get_logger().error('목표 위치가 유효하지 않거나 장애물입니다.')
            return False

        rx, ry = self.astar_planner.planning(sx, sy, gx, gy)
        if rx is None or ry is None:
            self.get_logger().info('경로를 찾을 수 없습니다.')
            return False

        self.path_x = rx
        self.path_y = ry
        self.path_index = 0
        self.path_planned = True
        self.get_logger().info(f'경로가 계획되었습니다. 길이: {len(rx)}')

        # 경로 시각화
        self.publish_path(rx, ry)

        return True

    def publish_path(self, rx, ry):
        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in zip(rx, ry):
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # 방향은 중요하지 않으므로 단위 쿼터니언 사용
            path_msg.poses.append(pose)
        self.path_publisher.publish(path_msg)
        self.get_logger().info('경로를 /planned_path 토픽으로 발행했습니다.')

    def control_loop(self):
        if self.astar_planner is None:
            self.get_logger().info('플래너 초기화 기다리는 중...')
            return

        if self.current_pose is None:
            self.get_logger().info('Odometry 데이터 기다리는 중...')
            return

        # 경로가 없으면 경로를 계획
        if not self.path_planned:
            path_planned = self.plan_path()
            if not path_planned:
                return

        # 경로를 따라감
        if self.path_index >= len(self.path_x):
            self.get_logger().info('목표 지점에 도달했습니다!')
            cmd_msg = Twist()
            self.cmd_vel_publisher.publish(cmd_msg)
            return

        goal_x = self.path_x[self.path_index]
        goal_y = self.path_y[self.path_index]

        # 목표 위치에 도달했는지 확인
        dx = goal_x - self.current_pose.position.x
        dy = goal_y - self.current_pose.position.y
        distance = math.hypot(dx, dy)

        if distance < 0.2:  # 임계값을 0.2로 설정
            self.get_logger().info(f'경로 점 {self.path_index}에 도달했습니다: ({goal_x:.2f}, {goal_y:.2f})')
            self.path_index += 1
            return

        # 로봇 제어
        angle_to_goal = math.atan2(dy, dx)

        # 현재 로봇의 방향을 얻음
        quaternion = (
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w)
        yaw = tf_transformations.euler_from_quaternion(quaternion)[2]

        angle_difference = self.normalize_angle(angle_to_goal - yaw)

        # 선속도와 각속도를 비례 제어
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.5 * distance
        cmd_msg.angular.z = 2.0 * angle_difference

        # 선속도와 각속도의 최대값을 제한
        max_linear_speed = 0.2
        max_angular_speed = 0.5

        if cmd_msg.linear.x > max_linear_speed:
            cmd_msg.linear.x = max_linear_speed

        if cmd_msg.angular.z > max_angular_speed:
            cmd_msg.angular.z = max_angular_speed
        elif cmd_msg.angular.z < -max_angular_speed:
            cmd_msg.angular.z = -max_angular_speed

        # 디버깅을 위한 로그 출력
        self.get_logger().info(f"현재 위치: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f})")
        self.get_logger().info(f"목표 위치: ({goal_x:.2f}, {goal_y:.2f})")
        self.get_logger().info(f"목표까지 거리: {distance:.2f}")
        self.get_logger().info(f"목표까지 각도: {angle_to_goal:.2f}")
        self.get_logger().info(f"현재 yaw: {yaw:.2f}")
        self.get_logger().info(f"각도 차이: {angle_difference:.2f}")
        self.get_logger().info(f"선속도: {cmd_msg.linear.x:.2f}")
        self.get_logger().info(f"각속도: {cmd_msg.angular.z:.2f}")

        self.cmd_vel_publisher.publish(cmd_msg)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = AStarNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
