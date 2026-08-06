# pose_locator_node.py
#
# YOLO-pose를 돌릴 만한 사양이 되는 머신에서 실행 (이 개발 PC가 아닐 수 있음
# - 이미지가 로컬 cv2.VideoCapture 대신 ROS2 토픽으로 들어오는 이유는
# camera_publisher.py 주석 참고).
#
# 카메라 프레임 하나가 들어올 때마다 하는 일:
#   1. 프레임에 대해 YOLO-pose 트래킹 실행 (사람마다 지속적인 트래커 id를
#      붙여줘서, 여러 프레임에 걸쳐 한 사람을 "락온"할 수 있게 함).
#   2. 이번 프레임에서 누구를 계속 따라갈지 고름 (vision_utils.select_person).
#   3. 그 사람의 발목 keypoint들을 뽑아서 하나의 "서 있는 픽셀"로 만듦
#      (vision_utils.extract_standing_pixel).
#   4. 그 픽셀을 캘리브레이션된 homography 행렬에 통과시켜서 map 프레임 기준
#      실제 (x, y)를 얻음 (vision_utils.apply_homography).
#   5. 그 결과를 geometry_msgs/PointStamped로 "person/position"에 publish함.
#      (person_tf_broadcaster_node가 이걸 구독해서 실제 TF로 바꿔줌 -
#      무거운 YOLO 추론과 다른 머신에서 돌 수 있도록, 예를 들면 Nav2 옆에서
#      돌 수 있도록 일부러 별도 노드로 분리해둠.)

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped
from ultralytics import YOLO

# 로컬 헬퍼 함수들 - 실제 계산/로직은 vision_utils.py 참고
from person_locator.vision_utils import (
    apply_homography,
    decode_jpeg,
    encode_jpeg,
    extract_standing_pixel,
    load_homography_yaml,
    select_person,
)


class PoseLocatorNode(Node):

    def __init__(self):
        super().__init__('pose_locator_node')

        # --- 파라미터, 전부 launch 파일이나 --ros-args -p로 override 가능 ---
        # 경로 없이 파일명만 주면 Ultralytics가 처음 실행할 때 자기 weights
        # 캐시로 자동 다운로드함; 이미 이 머신에 .pt 파일을 올려뒀으면
        # (예: /home/soo/corecode/COCO_WholeBody/yolov8n-pose.pt를 복사)
        # 절대 경로로 override
        self.declare_parameter('model_path', 'yolov8n-pose.pt')
        # 박스를 "사람"으로 볼 최소 YOLO 검출 신뢰도
        self.declare_parameter('conf_threshold', 0.5)
        # 발목 keypoint를 믿을 최소 신뢰도 - vision_utils.extract_standing_pixel로
        # 그대로 전달됨
        self.declare_parameter('ankle_conf_threshold', 0.5)
        # Ultralytics 내장 multi-object tracker 설정 - 검출된 사람마다 프레임
        # 간에 안정적인 id를 붙여줘서, select_person이 같은 사람을 계속
        # 따라갈 수 있게 해줌
        self.declare_parameter('tracker', 'bytetrack.yaml')
        # 캘리브레이션된 homography 행렬을 어디서 불러올지 -
        # `ros2 run person_locator calibrate_homography`로 생성됨
        self.declare_parameter('homography_yaml_path', 'config/person_homography.yaml')
        # 입력 이미지 토픽 (camera_publisher.py가 JPEG로 압축해서 publish하며,
        # 이 노드와 다른 머신일 수도 있음 - camera_publisher.py 주석 참고)
        self.declare_parameter('image_topic', 'camera/image_raw/compressed')
        # 출력 토픽: map 프레임 기준 사람의 위치
        self.declare_parameter('target_topic', 'person/position')
        # true면 박스+스켈레톤이 그려진 디버그용 이미지도 같이 publish함
        # - rqt_image_view로 검출이 잘 되는지 눈으로 확인할 수 있음.
        # 이것도 camera_publisher와 같은 이유로 JPEG 압축해서 내보냄
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('overlay_topic', 'person/debug_image/compressed')
        self.declare_parameter('overlay_jpeg_quality', 80)

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.ankle_conf_threshold = self.get_parameter('ankle_conf_threshold').value
        self.tracker_cfg = self.get_parameter('tracker').value
        homography_yaml_path = self.get_parameter('homography_yaml_path').value
        image_topic = self.get_parameter('image_topic').value
        target_topic = self.get_parameter('target_topic').value
        self.publish_overlay = self.get_parameter('publish_overlay').value
        overlay_topic = self.get_parameter('overlay_topic').value
        self.overlay_jpeg_quality = self.get_parameter('overlay_jpeg_quality').value

        # homography 행렬은 시작할 때 미리 로드함 - 파일이 없거나 형식이
        # 이상하면 나중에 조용히 이상한 좌표를 publish하는 대신
        # 시작 시점에 바로 실패하게 함
        try:
            self.homography = load_homography_yaml(homography_yaml_path)
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f'{homography_yaml_path}에서 homography를 불러오지 못함: {exc}. '
                f'먼저 `ros2 run person_locator calibrate_homography`를 실행하세요.'
            ) from exc

        self.get_logger().info(f'YOLO-pose 모델 로딩 중: {model_path}')
        self.model = YOLO(model_path)

        # QoS: 작고, reliable하고, 최신 것만 남기는 큐. 뭔가 잠깐 밀려도
        # 오래된 프레임/포인트가 쌓이길 원하는 게 아니라 항상 가장 최신
        # 것만 원함 - rc_car_chase의 webcam_locator_node와 동일한 설정
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.history = HistoryPolicy.KEEP_LAST

        self.image_sub = self.create_subscription(
            CompressedImage, image_topic, self.image_callback, qos
        )
        self.target_pub = self.create_publisher(PointStamped, target_topic, qos)
        self.overlay_pub = (
            self.create_publisher(CompressedImage, overlay_topic, 1)
            if self.publish_overlay else None
        )

        # 지금 "락온"해서 따라가고 있는 사람의 트래커 id
        # (vision_utils.select_person 참고) - None이면 아직 아무도 락온 안 한 것
        self.locked_track_id = None
        self.frame_count = 0

        self.get_logger().info(
            f'pose_locator_node: "{image_topic}" 구독, "{target_topic}"에 publish'
        )

    def image_callback(self, msg):
        # 들어온 JPEG 압축 메시지를 Ultralytics/OpenCV가 다룰 수 있는
        # OpenCV BGR numpy 프레임으로 압축 해제
        frame = decode_jpeg(msg)

        self.frame_count += 1
        if self.frame_count % 30 == 1:
            # 매 프레임마다 로그를 찍으면 너무 시끄러우니, 노드가 살아있고
            # 실제로 프레임을 받고 있는지 확인할 수 있게 주기적으로만 로그
            self.get_logger().info(f'pose_locator_node: {self.frame_count} 프레임 처리함')

        # 트래킹을 켠 채로 YOLO-pose 실행. `persist=True`는 Ultralytics한테
        # 트래킹 상태를 호출 사이(즉, 프레임 사이)에도 유지하라는 뜻이고,
        # 이게 있어야 같은 사람에 대해 .id가 안정적으로 유지됨.
        # verbose=False는 Ultralytics가 매 프레임마다 콘솔에 찍는 로그를 끔
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            tracker=self.tracker_cfg,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        # 이번 프레임에 어떤 사람을 따라갈지 결정
        index, track_id = select_person(boxes, self.locked_track_id)

        # 이번 프레임에 실제로 homography에 넣은 점/결과 - 검출이 없으면 None
        # 유지. 오버레이 디버그 이미지에 이 점을 표시할 때 씀 (아래 참고)
        standing_pixel = None
        world_xy = None

        if index is not None:
            # 다음 프레임의 select_person 호출이 같은 사람을 계속
            # 우선적으로 따라가도록 이 사람의 트래커 id를 기억해둠
            self.locked_track_id = track_id

            # 이 검출의 keypoint를 꺼냄. result.keypoints.xy는
            # (검출 개수, 17, 2) 형태고, result.keypoints.conf는
            # (검출 개수, 17) 형태인데 모델 설정에 따라 None일 수도 있음
            keypoints_xy = result.keypoints.xy[index].cpu().numpy()
            keypoints_conf = (
                result.keypoints.conf[index].cpu().numpy()
                if result.keypoints.conf is not None
                else None
            )
            box_xyxy = boxes.xyxy[index].tolist()

            # 발목 keypoint(또는 bbox fallback)를 이미지 좌표계의
            # "서 있는 픽셀" (u, v) 하나로 변환
            u, v = extract_standing_pixel(
                keypoints_xy, keypoints_conf, box_xyxy, self.ankle_conf_threshold
            )

            # 그 픽셀을 캘리브레이션된 homography에 통과시켜서
            # map 프레임 기준 실제 (x, y)(미터 단위)를 얻음
            x, y = apply_homography(u, v, self.homography)
            standing_pixel = (u, v)
            world_xy = (x, y)

            # 결과를 map 프레임 기준 PointStamped로 publish
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = 'map'
            point_msg.point.x = x
            point_msg.point.y = y
            point_msg.point.z = 0.0
            self.target_pub.publish(point_msg)
        else:
            # 이번 프레임에 아무도 검출 안 됨 - 다음에 검출되는 사람이
            # 누구든 새로 잡을 수 있도록 락을 풀고, 오래된/마지막으로
            # 알던 위치는 일부러 publish하지 않음
            self.locked_track_id = None

        if self.publish_overlay:
            # result.plot()이 박스+스켈레톤 keypoint를 프레임 복사본에
            # 그려줌 - `ros2 run rqt_image_view rqt_image_view`로 검출/트래킹이
            # 잘 되는지 눈으로 확인할 때 유용함
            overlay = result.plot()

            # homography에 실제로 들어간 점(양발 중점)과 그 결과 map 좌표를
            # 자홍색으로 따로 표시 - YOLO 스켈레톤 색과 안 겹치게 해서, 캘리브레이션이
            # 맞는지(마커가 실제 발 위치에 찍히는지) 눈으로 바로 확인할 수 있게 함
            if standing_pixel is not None:
                u, v = standing_pixel
                x, y = world_xy
                cv2.drawMarker(overlay, (int(u), int(v)), (255, 0, 255),
                                cv2.MARKER_CROSS, 20, 3)
                cv2.putText(overlay, f'map: ({x:.2f}, {y:.2f})', (int(u) + 12, int(v) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            overlay_msg = encode_jpeg(overlay, quality=self.overlay_jpeg_quality)
            self.overlay_pub.publish(overlay_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseLocatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
