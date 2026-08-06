# log_amr_positions.py
#
# pick_pixel_points.py(웹캠 PC용)의 짝. 이 PC(AMR을 teleop으로 모는 쪽)에서
# 돌려서, AMR을 스티커 지점에 세울 때마다 그 순간의 amcl_pose(x, y)를
# 순서대로 기록함. pick_pixel_points.py 쪽에서 같은 순서로 픽셀을 클릭해두면
# compute_homography_from_points.py로 두 파일을 순서대로 짝지어서
# homography를 계산할 수 있음.
#
# 사용법:
#   ros2 run person_locator log_amr_positions --ros-args -p output:=world_points.yaml
#
# AMR을 스티커 지점에 세우고, (웹캠 쪽 담당자가 그 지점을 클릭했다고 확인해준
# 뒤) 이 터미널에서 Enter를 누르면 그 순간의 최신 amcl_pose가 기록됨.
# 'u' + Enter = 마지막 점 취소, 'q' + Enter = 저장 후 종료.

import os
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
import yaml


class PoseListener(Node):
    """amcl_pose를 계속 구독해서 최신 (x, y)만 들고 있음 - main()의 입력
    루프가 Enter를 받을 때마다 이 값의 스냅샷을 찍어감."""

    def __init__(self):
        super().__init__('log_amr_positions')

        self.declare_parameter('pose_topic', '/amcl_pose')
        self.declare_parameter('output', 'world_points.yaml')

        pose_topic = self.get_parameter('pose_topic').value
        self.output_path = self.get_parameter('output').value

        self.latest_xy = None
        # amcl_pose는 TRANSIENT_LOCAL로 발행됨(마지막 값을 새 구독자에게 즉시
        # 재전달) - 기본 QoS(VOLATILE)로 구독하면 이 재전달을 못 받아서, 로봇이
        # 안 움직여 amcl이 새로 publish를 안 하는 동안은 계속 기다리게 됨
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped, pose_topic, self.pose_callback, qos
        )
        self.get_logger().info(f'log_amr_positions: "{pose_topic}"에서 pose 기다리는 중...')

    def pose_callback(self, msg):
        self.latest_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)


def main(args=None):
    rclpy.init(args=args)
    node = PoseListener()

    # rclpy.spin을 별도 스레드에서 계속 돌려서 latest_xy가 항상 최신으로
    # 갱신되게 하고, 메인 스레드는 input()으로 Enter 입력만 기다림
    # (input()이 블로킹이라 같은 스레드에서 spin까지 하면 안 됨)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('amcl_pose에서 첫 메시지를 기다리는 중...')
    while node.latest_xy is None and rclpy.ok():
        pass

    world_points = []
    print('AMR을 스티커 지점에 세운 뒤 Enter를 누르면 현재 위치가 기록됩니다.')
    print("(웹캠 쪽에서 같은 지점을 픽셀로 클릭한 것과 순서가 반드시 일치해야 함)")
    print("'u' + Enter = 마지막 점 취소, 'q' + Enter = 저장 후 종료")

    try:
        while rclpy.ok():
            raw = input(f'[{len(world_points) + 1}번째 점] Enter=기록 / u=취소 / q=종료: ').strip()

            if raw == 'q':
                break
            if raw == 'u':
                if world_points:
                    removed = world_points.pop()
                    print(f'취소됨: {removed}')
                continue

            x, y = node.latest_xy
            world_points.append((x, y))
            print(f'[{len(world_points)}] world=({x:.3f}, {y:.3f}) 기록됨')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if world_points:
            output_path = node.output_path
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.safe_dump(
                    {'world_points': [[float(x), float(y)] for x, y in world_points]},
                    f, sort_keys=False,
                )
            print(f'실좌표 {len(world_points)}개를 {output_path}에 저장했습니다.')
        else:
            print('기록된 점이 없어 저장하지 않습니다.')

        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
