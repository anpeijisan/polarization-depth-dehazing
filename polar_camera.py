"""PHX050S-P 实时四角度采集。只在内存中保留最新一帧。"""
from dataclasses import dataclass, replace
from collections import deque
from queue import Empty, Full, Queue
import ctypes
import math
import threading
import time

import numpy as np

PIXEL_FORMAT = "PolarizedAngles_0d_45d_90d_135d_Mono8"
PIXEL_FORMAT_ID = 0x8220000F
ANGLES = (0, 45, 90, 135)


@dataclass(frozen=True)
class PolarFrame:
    """同一次采集的四通道数据；angles 为只读 H×W×4 uint8，顺序 0/45/90/135。"""
    angles: np.ndarray
    frame_id: int
    camera_timestamp_ns: int
    host_received_ns: int
    payload_bytes: int = 0
    copy_ms: float = 0.0
    software_trigger_to_receive_ms: float = None
    host_ready_ns: int = 0
    software_trigger_to_ready_ms: float = None
    offset_x: int = 0
    offset_y: int = 0

    def angle(self, degrees: int) -> np.ndarray:
        return self.angles[:, :, ANGLES.index(degrees)]


def copy_angle_buffer(buffer, host_received_ns=None, trigger_started_ns=None) -> PolarFrame:
    """在归还 Arena 缓冲区前复制有效像素，保留各通道原始数值。"""
    if host_received_ns is None:
        host_received_ns = time.perf_counter_ns()
    copy_started_ns = time.perf_counter_ns()
    if buffer.is_incomplete:
        raise ValueError("The camera returned an incomplete image; the frame was discarded.")
    pixel_format = getattr(buffer.pixel_format, "value", buffer.pixel_format)
    if int(pixel_format) != PIXEL_FORMAT_ID:
        raise ValueError(f"Unexpected image format: {pixel_format}")
    width, height = int(buffer.width), int(buffer.height)
    row_bytes = width * 4
    stride = row_bytes + int(buffer.padding_x)
    required = stride * height
    if buffer.size_filled < required:
        raise ValueError(f"Insufficient image data: {buffer.size_filled} < {required}")
    address = ctypes.cast(buffer.pdata, ctypes.c_void_p).value
    view = np.ctypeslib.as_array((ctypes.c_uint8 * required).from_address(address))
    angles = view.reshape(height, stride)[:, :row_bytes].copy().reshape(height, width, 4)
    angles.flags.writeable = False
    copy_ms = (time.perf_counter_ns() - copy_started_ns) / 1_000_000
    trigger_ms = ((host_received_ns - trigger_started_ns) / 1_000_000
                  if trigger_started_ns is not None else None)
    return PolarFrame(angles, int(buffer.frame_id), int(buffer.timestamp_ns),
                      host_received_ns, int(buffer.size_filled), copy_ms, trigger_ms)


class PolarCamera:
    """线程化采集接口；get_latest 返回最新完整帧，不保存图像。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._thread = None
        self._stop = threading.Event()
        self._commands = Queue()
        self.status = "Not started"
        self.error = None
        self.exposure_us = None
        self.exposure_range = None
        self.received = 0
        self.incomplete = 0
        self.fps = 0.0
        self.device_name = ""
        self.active_roi = None
        self._recent = deque(maxlen=300)
        self._trigger_times = deque(maxlen=300)
        self.scheduled = self.triggered = self.skipped_busy = self.skipped_unarmed = 0
        self.delivery_dropped = 0
        self._delivery = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, mode="continuous", exposure_us=2000.0, interval_ms=250, trigger_line="Line0", serial="", roi=None):
        if self.running:
            raise RuntimeError("Acquisition is already running.")
        if mode not in ("continuous", "software", "simulated", "external"):
            raise ValueError("Acquisition mode must be continuous, software, simulated, or external.")
        if exposure_us <= 0 or interval_ms <= 0:
            raise ValueError("Exposure time and trigger interval must be positive.")
        roi = (0, 0, 2448, 2048) if roi is None else tuple(roi)
        if (len(roi) != 4 or any(not isinstance(value, int) for value in roi)
                or roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0
                or roi[0] + roi[2] > 2448 or roi[1] + roi[3] > 2048):
            raise ValueError("The acquisition region must fit within the 2448×2048 sensor and have positive dimensions.")
        self._stop.clear()
        self._commands = Queue()
        if self._delivery is not None:
            self._delivery = Queue(maxsize=self._delivery.maxsize)
        with self._lock:
            self._latest = None
            self.error = None
            self.status = "Connecting to camera"
            self.received = self.incomplete = 0
            self.fps = 0.0
            self.active_roi = None
            self._recent.clear()
            self._trigger_times.clear()
            self.scheduled = self.triggered = self.skipped_busy = self.skipped_unarmed = 0
            self.delivery_dropped = 0
        self._thread = threading.Thread(target=self._run, args=(mode, float(exposure_us), int(interval_ms), trigger_line, serial.strip(), roi), daemon=False)
        self._thread.start()

    def set_exposure(self, exposure_us):
        if not self.running:
            raise RuntimeError("Start acquisition first.")
        self._commands.put(float(exposure_us))

    def get_latest(self):
        with self._lock:
            return self._latest

    def enable_delivery(self, capacity=4):
        """启动前启用有界逐帧队列；下游应持续消费并检查 delivery_dropped。"""
        if self.running:
            raise RuntimeError("Configure the delivery queue before starting acquisition.")
        if capacity < 1:
            raise ValueError("Queue capacity must be positive.")
        self._delivery = Queue(maxsize=capacity)

    def get_next(self, timeout=None):
        if self._delivery is None:
            raise RuntimeError("Call enable_delivery first.")
        return self._delivery.get(timeout=timeout)

    def reset_metrics(self):
        with self._lock:
            self._recent.clear()
            self._trigger_times.clear()

    def get_metrics(self):
        """仅用主机单调时钟统计最近接收的完整帧；不读取或保存图像。"""
        with self._lock:
            recent = tuple(self._recent)
            triggers = tuple(self._trigger_times)
        if not recent:
            return None
        latest_ns = recent[-1][0]
        recent = tuple(f for f in recent if latest_ns - f[0] <= 5_000_000_000)
        span_s = (recent[-1][0] - recent[0][0]) / 1e9
        fps = (len(recent) - 1) / span_s if span_s > 0 else 0.0
        mbps = sum(f[1] for f in recent[1:]) * 8 / 1e6 / span_s if span_s > 0 else 0.0
        copy_values = [f[2] for f in recent]
        trigger_values = [f[3] for f in recent if f[3] is not None]
        ready_values = [f[4] for f in recent if f[4] is not None]
        intervals = np.diff(triggers) / 1e6 if len(triggers) >= 2 else []
        return {"trigger_interval_mean_ms": float(np.mean(intervals)) if len(intervals) else None,
                "trigger_interval_p95_ms": float(np.percentile(intervals, 95)) if len(intervals) else None,
                "ready_p95_ms": float(np.percentile(ready_values, 95)) if ready_values else None,
                "fps": fps, "payload_mbps": mbps, "payload_bytes": recent[-1][1],
                "copy_mean_ms": float(np.mean(copy_values)),
                "trigger_latest_ms": trigger_values[-1] if trigger_values else None,
                "trigger_p95_ms": float(np.percentile(trigger_values, 95)) if trigger_values else None,
                "samples": len(recent)}

    def stop(self, timeout=5):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("The camera thread is still running. Wait for device operations to finish.")

    def _run(self, mode, exposure_us, interval_ms, trigger_line, serial, roi):
        device = None
        stream_started = False
        original = {}
        try:
            from arena_api.system import system
            infos = [info for info in system.device_infos if str(info.get("model", "")).startswith("PHX050S-P")]
            if serial:
                infos = [info for info in infos if str(info.get("serial")) == serial]
            if not infos:
                raise RuntimeError("No PHX050S-P camera found. Check power, Ethernet, IP settings, and whether another application is using it.")
            if len(infos) > 1:
                raise RuntimeError("Multiple PHX050S-P cameras found. Specify the camera serial number.")
            info = infos[0]
            try:
                device = system.create_device([info])[0]
            except Exception as exc:
                if "INVALID_ADDRESS" not in str(exc):
                    raise
                raise RuntimeError(
                    f"Cannot open the camera (SDK address error -1015).\nCamera IP: {info['ip']}; "
                    f"subnet mask: {info['subnetmask']}.\n"
                    "Use the LUCID IP configuration tool to check that the camera and Ethernet adapter share a subnet, "
                    "and verify that the camera IP does not conflict with another device.\n"
                    "After adjusting the settings, close other camera applications and restart acquisition."
                ) from exc
            self.device_name = f"{info['model']} / {info['serial']}"
            nodes = device.nodemap
            # 修改节点前保存当前值，以便停止采集后恢复。
            original = {name: nodes[name].value for name in ("PixelFormat", "ExposureAuto", "ExposureTime", "TriggerSelector", "Width", "Height", "OffsetX", "OffsetY")}
            nodes["TriggerSelector"].value = "FrameStart"
            original.update({name: nodes[name].value for name in ("TriggerMode", "TriggerSource", "TriggerActivation")})
            nodes["TriggerMode"].value = "Off"
            nodes["PixelFormat"].value = PIXEL_FORMAT
            nodes["OffsetX"].value = 0
            nodes["OffsetY"].value = 0
            for name, value in (("Width", roi[2]), ("Height", roi[3]),
                                ("OffsetX", roi[0]), ("OffsetY", roi[1])):
                node = nodes[name]
                increment = getattr(node, "inc", None)
                if value < node.min or value > node.max or (increment and (value - node.min) % increment):
                    raise ValueError(f"{name}={value} is outside the camera range {node.min}–{node.max} with increment {increment}.")
                node.value = value
            self.active_roi = (int(nodes["OffsetX"].value), int(nodes["OffsetY"].value),
                               int(nodes["Width"].value), int(nodes["Height"].value))
            nodes["ExposureAuto"].value = "Off"
            self.exposure_range = (float(nodes["ExposureTime"].min), float(nodes["ExposureTime"].max))
            self._apply_exposure(nodes, exposure_us)
            nodes["TriggerSelector"].value = "FrameStart"
            if mode != "continuous":
                nodes["TriggerSource"].value = "Software" if mode in ("software", "simulated") else trigger_line
                if mode == "external":
                    nodes["TriggerActivation"].value = "RisingEdge"
                nodes["TriggerMode"].value = "On"
            stream = device.tl_stream_nodemap
            stream["StreamAutoNegotiatePacketSize"].value = True
            stream["StreamPacketResendEnable"].value = True
            stream["StreamBufferHandlingMode"].value = "NewestOnly"
            device.start_stream()
            stream_started = True
            self.status = "Waiting for external trigger" if mode == "external" else "Acquiring"
            next_trigger = time.perf_counter()
            pending_trigger_ns = None
            fps_start = next_trigger
            fps_count = 0
            while not self._stop.is_set():
                try:
                    value = self._commands.get_nowait()
                except Empty:
                    pass
                else:
                    try:
                        self._apply_exposure(nodes, value)
                    except Exception as exc:
                        self.error = f"Exposure adjustment failed: {exc}"
                timed = mode in ("software", "simulated")
                now = time.perf_counter()
                if timed and now >= next_trigger:
                    # 固定时间轴；迟到时记录错过的节拍，不突发补拍。
                    slots = int((now - next_trigger) / (interval_ms / 1000.0)) + 1
                    self.scheduled += slots
                    self.skipped_busy += slots - 1
                    next_trigger += slots * interval_ms / 1000.0
                    if pending_trigger_ns is not None:
                        self.skipped_busy += 1
                    elif not bool(nodes["TriggerArmed"].value):
                        self.skipped_unarmed += 1
                    else:
                        pending_trigger_ns = time.perf_counter_ns()
                        nodes["TriggerSoftware"].execute()
                        self.triggered += 1
                        with self._lock:
                            self._trigger_times.append(pending_trigger_ns)
                if timed and pending_trigger_ns is None:
                    self._stop.wait(max(0, next_trigger - time.perf_counter()))
                    continue
                try:
                    timeout = 100
                    if timed:
                        timeout = max(1, min(100, math.ceil((next_trigger - time.perf_counter()) * 1000)))
                    buffer = device.get_buffer(timeout=timeout)
                    received_ns = time.perf_counter_ns()
                except TimeoutError:
                    continue
                try:
                    if buffer.is_incomplete:
                        self.incomplete += 1
                        pending_trigger_ns = None
                        continue
                    trigger_started_ns = pending_trigger_ns
                    frame = copy_angle_buffer(buffer, received_ns, trigger_started_ns)
                    pending_trigger_ns = None
                finally:
                    device.requeue_buffer(buffer)
                with self._lock:
                    ready_ns = time.perf_counter_ns()
                    trigger_ready_ms = ((ready_ns - trigger_started_ns) / 1_000_000
                                        if trigger_started_ns is not None else None)
                    frame = replace(frame, host_ready_ns=ready_ns,
                                    software_trigger_to_ready_ms=trigger_ready_ms,
                                    offset_x=self.active_roi[0], offset_y=self.active_roi[1])
                    self._latest = frame
                    self._recent.append((frame.host_received_ns, frame.payload_bytes,
                                         frame.copy_ms, frame.software_trigger_to_receive_ms,
                                         frame.software_trigger_to_ready_ms))
                    self.received += 1
                if self._delivery is not None:
                    try:
                        self._delivery.put_nowait(frame)
                    except Full:
                        self.delivery_dropped += 1
                fps_count += 1
                now = time.perf_counter()
                if now - fps_start >= 1.0:
                    self.fps = fps_count / (now - fps_start)
                    fps_start, fps_count = now, 0
        except Exception as exc:
            self.error = str(exc)
            self.status = "Acquisition failed"
        finally:
            if device is not None:
                try:
                    if stream_started:
                        device.stop_stream()
                    if original:
                        nodes["TriggerMode"].value = "Off"
                        nodes["OffsetX"].value = 0
                        nodes["OffsetY"].value = 0
                        nodes["Width"].value = original["Width"]
                        nodes["Height"].value = original["Height"]
                        nodes["OffsetX"].value = original["OffsetX"]
                        nodes["OffsetY"].value = original["OffsetY"]
                        nodes["PixelFormat"].value = original["PixelFormat"]
                        nodes["ExposureTime"].value = original["ExposureTime"]
                        nodes["ExposureAuto"].value = original["ExposureAuto"]
                        nodes["TriggerSource"].value = original["TriggerSource"]
                        nodes["TriggerActivation"].value = original["TriggerActivation"]
                        nodes["TriggerMode"].value = original["TriggerMode"]
                        nodes["TriggerSelector"].value = original["TriggerSelector"]
                except Exception as exc:
                    self.error = f"{self.error + '; ' if self.error else ''}Failed to restore camera settings: {exc}"
                finally:
                    try:
                        system.destroy_device(device)
                    except Exception as exc:
                        self.error = f"{self.error + '; ' if self.error else ''}Failed to release camera: {exc}"
            if self.status != "Acquisition failed":
                self.status = "Stopped"

    def _apply_exposure(self, nodes, value):
        minimum, maximum = self.exposure_range
        if not minimum <= value <= maximum:
            raise ValueError(f"Exposure must be within {minimum:g}–{maximum:g} μs.")
        nodes["ExposureTime"].value = value
        self.exposure_us = float(nodes["ExposureTime"].value)


