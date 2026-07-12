import json
import socket

import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883

camera_name = "cam01"
pc_name = socket.gethostname()


class Publisher:
    def __init__(
        self,
        broker_host: str = BROKER_HOST,
        broker_port: int = BROKER_PORT,
        camera_name: str = camera_name,
        pc_name: str = pc_name,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.camera_name = camera_name
        self.pc_name = pc_name
        self.pub_topic = f"camera/{self.camera_name}"
        self.pub_client = mqtt.Client(
            client_id=f"{self.pc_name}_{self.camera_name}_publisher"
        )

    def publish_detections(self, frame_id, timestamp, detections):
        """Publish raw detections (no track ids).

        detections: {"point": [{"x","y","score"}, ...], "bbox": [...]}
        Coordinates are in the edge sensor frame (e.g. FHD crop).
        """
        payload = {
            "camera_name": self.camera_name,
            "pc_name": self.pc_name,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "detections": detections,
        }
        self.pub_client.publish(
            self.pub_topic,
            json.dumps(payload),
            qos=0,
            retain=False,
        )

    def publish_result(self, frame_id, timestamp, objects):
        """Legacy: publish already-tracked objects (kept for compatibility)."""
        payload = {
            "camera_name": self.camera_name,
            "pc_name": self.pc_name,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "objects": objects,
        }
        self.pub_client.publish(
            self.pub_topic,
            json.dumps(payload),
            qos=0,
            retain=False,
        )

    def start(self):
        self.pub_client.connect(self.broker_host, self.broker_port, keepalive=60)
        self.pub_client.loop_start()
        print(f"[PUB] started camera_name={self.camera_name}")

    def stop(self):
        self.pub_client.loop_stop()
        self.pub_client.disconnect()
        print(f"[PUB] stopped camera_name={self.camera_name}")
