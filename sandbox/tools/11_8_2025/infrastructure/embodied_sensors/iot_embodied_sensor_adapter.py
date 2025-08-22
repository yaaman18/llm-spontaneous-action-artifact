"""
IoT身体感覚センサーアダプター

エナクティブ認知理論の実装において、実際のハードウェアセンサーから
身体感覚データを取得し、仮想的な身体化認知を実現する。

対応センサー：
- IMU（慣性計測装置）：固有受容感覚・前庭感覚
- 生体センサー：内受容感覚
- 空間センサー：環境との身体的関係
- ウェアラブルデバイス：継続的な身体状態監視

理論的基盤：
- Ezequiel Di Paolo のエナクティブ認知
- 身体図式の動的更新
- 感覚運動結合による記憶錨定
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import numpy as np
from enum import Enum

# IoTライブラリのシミュレーション（実装時は実際のライブラリを使用）
try:
    # 実際の実装では以下のようなライブラリを使用
    # import paho.mqtt.client as mqtt
    # import bluetooth
    # import serial
    # from bleak import BleakClient, BleakScanner
    pass
except ImportError:
    pass


class SensorType(Enum):
    """センサータイプの定義"""
    IMU = "imu"
    HEART_RATE = "heart_rate"
    SKIN_CONDUCTANCE = "skin_conductance"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    PROXIMITY = "proximity"
    FORCE = "force"


class ConnectionProtocol(Enum):
    """接続プロトコルの定義"""
    BLUETOOTH_LE = "bluetooth_le"
    WIFI = "wifi"
    ZIGBEE = "zigbee"
    MQTT = "mqtt"
    SERIAL = "serial"
    HTTP_REST = "http_rest"
    WEBSOCKET = "websocket"


@dataclass
class SensorReading:
    """センサー読み取り値"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime
    raw_data: Dict[str, Any]
    processed_data: Dict[str, float]
    confidence: float
    calibration_status: str


@dataclass
class SensorConfiguration:
    """センサー設定"""
    sensor_id: str
    sensor_type: SensorType
    connection_protocol: ConnectionProtocol
    connection_params: Dict[str, Any]
    sampling_rate: float  # Hz
    calibration_params: Dict[str, Any]
    data_processing_pipeline: List[str]


class IoTSensorInterface(ABC):
    """IoTセンサーインターフェース"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """センサーへの接続"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """センサーからの切断"""
        pass
    
    @abstractmethod
    async def read_sensor_data(self) -> Optional[SensorReading]:
        """センサーデータの読み取り"""
        pass
    
    @abstractmethod
    async def calibrate_sensor(self) -> bool:
        """センサーの校正"""
        pass
    
    @abstractmethod
    def get_sensor_status(self) -> Dict[str, Any]:
        """センサー状態の取得"""
        pass


class IMUSensorAdapter(IoTSensorInterface):
    """IMU（慣性計測装置）センサーアダプター"""
    
    def __init__(self, config: SensorConfiguration):
        self.config = config
        self.connection = None
        self.is_connected = False
        self.calibration_data = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """IMUセンサーへの接続"""
        try:
            # 実際の実装では、具体的なIMUセンサーのAPIを使用
            # 例：MPU-6050, LSM9DS1, etc.
            
            if self.config.connection_protocol == ConnectionProtocol.BLUETOOTH_LE:
                return await self._connect_bluetooth_le()
            elif self.config.connection_protocol == ConnectionProtocol.SERIAL:
                return await self._connect_serial()
            else:
                # その他のプロトコル対応
                return await self._connect_generic()
                
        except Exception as e:
            self.logger.error(f"IMU connection failed: {e}")
            return False
    
    async def _connect_bluetooth_le(self) -> bool:
        """Bluetooth LE接続（シミュレーション）"""
        # 実際の実装例：
        # from bleak import BleakClient
        # self.connection = BleakClient(self.config.connection_params['device_address'])
        # await self.connection.connect()
        
        await asyncio.sleep(0.1)  # 接続シミュレーション
        self.is_connected = True
        self.logger.info(f"IMU sensor {self.config.sensor_id} connected via Bluetooth LE")
        return True
    
    async def _connect_serial(self) -> bool:
        """シリアル接続（シミュレーション）"""
        # 実際の実装例：
        # import serial
        # self.connection = serial.Serial(
        #     port=self.config.connection_params['port'],
        #     baudrate=self.config.connection_params['baudrate']
        # )
        
        await asyncio.sleep(0.1)
        self.is_connected = True
        self.logger.info(f"IMU sensor {self.config.sensor_id} connected via Serial")
        return True
    
    async def _connect_generic(self) -> bool:
        """汎用接続"""
        await asyncio.sleep(0.1)
        self.is_connected = True
        return True
    
    async def disconnect(self) -> bool:
        """センサーからの切断"""
        if self.connection:
            # await self.connection.disconnect()
            pass
        self.is_connected = False
        self.logger.info(f"IMU sensor {self.config.sensor_id} disconnected")
        return True
    
    async def read_sensor_data(self) -> Optional[SensorReading]:
        """IMUデータの読み取り"""
        if not self.is_connected:
            return None
        
        try:
            # 実際の実装では、センサーから実データを取得
            raw_data = await self._read_raw_imu_data()
            processed_data = self._process_imu_data(raw_data)
            
            return SensorReading(
                sensor_id=self.config.sensor_id,
                sensor_type=self.config.sensor_type,
                timestamp=datetime.now(),
                raw_data=raw_data,
                processed_data=processed_data,
                confidence=self._calculate_reading_confidence(raw_data),
                calibration_status="calibrated"
            )
            
        except Exception as e:
            self.logger.error(f"IMU data reading failed: {e}")
            return None
    
    async def _read_raw_imu_data(self) -> Dict[str, Any]:
        """生IMUデータの読み取り（シミュレーション）"""
        # 実際の実装では、センサーAPIを使用
        # 例：accelerometer, gyroscope, magnetometerの値を取得
        
        # シミュレーションデータ（現実的な値）
        return {
            'accelerometer': {
                'x': np.random.normal(0, 0.1),  # m/s²
                'y': np.random.normal(0, 0.1),
                'z': np.random.normal(-9.8, 0.2)  # 重力方向
            },
            'gyroscope': {
                'x': np.random.normal(0, 0.05),  # rad/s
                'y': np.random.normal(0, 0.05),
                'z': np.random.normal(0, 0.05)
            },
            'magnetometer': {
                'x': np.random.normal(25, 5),    # μT
                'y': np.random.normal(-10, 3),
                'z': np.random.normal(40, 8)
            },
            'temperature': np.random.normal(25, 2)  # ℃
        }
    
    def _process_imu_data(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """IMUデータの処理"""
        processed = {}
        
        # 加速度データの処理
        acc = raw_data['accelerometer']
        processed['gravity_x'] = acc['x']
        processed['gravity_y'] = acc['y']
        processed['gravity_z'] = acc['z']
        processed['total_acceleration'] = np.sqrt(acc['x']**2 + acc['y']**2 + acc['z']**2)
        
        # ジャイロスコープデータの処理
        gyro = raw_data['gyroscope']
        processed['angular_velocity_x'] = gyro['x']
        processed['angular_velocity_y'] = gyro['y']
        processed['angular_velocity_z'] = gyro['z']
        
        # 姿勢推定（簡易版）
        processed['pitch'] = np.arctan2(acc['x'], np.sqrt(acc['y']**2 + acc['z']**2))
        processed['roll'] = np.arctan2(acc['y'], acc['z'])
        
        # 平衡状態の推定
        gravity_deviation = abs(processed['total_acceleration'] - 9.8)
        processed['balance_state'] = max(0.0, 1.0 - gravity_deviation / 2.0)
        
        # 空間安定性の推定
        angular_velocity_magnitude = np.sqrt(gyro['x']**2 + gyro['y']**2 + gyro['z']**2)
        processed['spatial_stability'] = max(0.0, 1.0 - angular_velocity_magnitude)
        
        # 時間整合性（前回の読み取りとの比較）
        processed['temporal_coherence'] = 0.8  # 簡易実装
        
        return processed
    
    def _calculate_reading_confidence(self, raw_data: Dict[str, Any]) -> float:
        """読み取り信頼度の計算"""
        # センサーノイズ、温度、キャリブレーション状態等を考慮
        base_confidence = 0.9
        
        # 温度による補正
        temp = raw_data.get('temperature', 25)
        temp_factor = max(0.8, 1.0 - abs(temp - 25) / 50)
        
        return base_confidence * temp_factor
    
    async def calibrate_sensor(self) -> bool:
        """IMUセンサーの校正"""
        if not self.is_connected:
            return False
        
        self.logger.info(f"Calibrating IMU sensor {self.config.sensor_id}")
        
        # 校正データの収集
        calibration_samples = []
        for _ in range(100):
            raw_data = await self._read_raw_imu_data()
            calibration_samples.append(raw_data)
            await asyncio.sleep(0.01)
        
        # 校正パラメータの計算
        self.calibration_data = self._calculate_calibration_parameters(calibration_samples)
        
        self.logger.info(f"IMU sensor {self.config.sensor_id} calibration completed")
        return True
    
    def _calculate_calibration_parameters(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """校正パラメータの計算"""
        # 各軸のオフセットとスケールファクターを計算
        acc_x_values = [s['accelerometer']['x'] for s in samples]
        acc_y_values = [s['accelerometer']['y'] for s in samples]
        acc_z_values = [s['accelerometer']['z'] for s in samples]
        
        return {
            'acc_x_offset': np.mean(acc_x_values),
            'acc_y_offset': np.mean(acc_y_values),
            'acc_z_offset': np.mean(acc_z_values) + 9.8,  # 重力補正
            'gyro_x_offset': np.mean([s['gyroscope']['x'] for s in samples]),
            'gyro_y_offset': np.mean([s['gyroscope']['y'] for s in samples]),
            'gyro_z_offset': np.mean([s['gyroscope']['z'] for s in samples])
        }
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """センサー状態の取得"""
        return {
            'sensor_id': self.config.sensor_id,
            'is_connected': self.is_connected,
            'calibration_status': 'calibrated' if self.calibration_data else 'not_calibrated',
            'sampling_rate': self.config.sampling_rate,
            'last_reading': datetime.now().isoformat()
        }


class BiometricSensorAdapter(IoTSensorInterface):
    """生体センサーアダプター"""
    
    def __init__(self, config: SensorConfiguration):
        self.config = config
        self.connection = None
        self.is_connected = False
        self.baseline_values = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """生体センサーへの接続"""
        try:
            # 心拍センサー、皮膚電導度センサー等への接続
            await asyncio.sleep(0.1)
            self.is_connected = True
            self.logger.info(f"Biometric sensor {self.config.sensor_id} connected")
            return True
        except Exception as e:
            self.logger.error(f"Biometric sensor connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """センサーからの切断"""
        self.is_connected = False
        self.logger.info(f"Biometric sensor {self.config.sensor_id} disconnected")
        return True
    
    async def read_sensor_data(self) -> Optional[SensorReading]:
        """生体データの読み取り"""
        if not self.is_connected:
            return None
        
        try:
            raw_data = await self._read_raw_biometric_data()
            processed_data = self._process_biometric_data(raw_data)
            
            return SensorReading(
                sensor_id=self.config.sensor_id,
                sensor_type=self.config.sensor_type,
                timestamp=datetime.now(),
                raw_data=raw_data,
                processed_data=processed_data,
                confidence=self._calculate_biometric_confidence(raw_data),
                calibration_status="calibrated"
            )
            
        except Exception as e:
            self.logger.error(f"Biometric data reading failed: {e}")
            return None
    
    async def _read_raw_biometric_data(self) -> Dict[str, Any]:
        """生体データの読み取り（シミュレーション）"""
        return {
            'heart_rate': np.random.normal(72, 8),           # bpm
            'respiratory_rate': np.random.normal(16, 2),     # breaths/min
            'skin_conductance': np.random.uniform(0.1, 1.0), # μS
            'skin_temperature': np.random.normal(32, 1),     # ℃
            'blood_oxygen': np.random.normal(98, 1)          # SpO2 %
        }
    
    def _process_biometric_data(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """生体データの処理"""
        processed = {}
        
        # 心拍関連
        hr = raw_data['heart_rate']
        processed['heart_rate'] = hr
        processed['heart_rate_normalized'] = (hr - 60) / 40  # 60-100bpmを0-1に正規化
        
        # 呼吸関連
        rr = raw_data['respiratory_rate']
        processed['respiratory_rate'] = rr
        processed['respiratory_rate_normalized'] = (rr - 12) / 8  # 12-20を0-1に正規化
        
        # 皮膚電導度（覚醒度の指標）
        sc = raw_data['skin_conductance']
        processed['skin_conductance'] = sc
        processed['arousal_level'] = min(1.0, sc)  # 覚醒度として使用
        
        # 情動価の推定（心拍変動と皮膚電導度から）
        hr_variability = abs(hr - 72) / 72  # 正常値からの偏差
        emotional_arousal = sc
        
        # 簡易的な情動価推定
        if hr_variability < 0.1 and emotional_arousal < 0.5:
            processed['emotional_valence'] = 0.2  # 穏やか（正の情動）
        elif hr_variability > 0.2 or emotional_arousal > 0.8:
            processed['emotional_valence'] = -0.3  # ストレス（負の情動）
        else:
            processed['emotional_valence'] = 0.0  # 中性
        
        # 内臓感覚記憶強度（自律神経活動から推定）
        autonomic_activity = (hr_variability + emotional_arousal) / 2
        processed['visceral_memory_strength'] = min(1.0, autonomic_activity + 0.3)
        
        # 内受容感覚認識度（個体差があるため基本値）
        processed['interoceptive_awareness'] = 0.6
        
        return processed
    
    def _calculate_biometric_confidence(self, raw_data: Dict[str, Any]) -> float:
        """生体データの信頼度計算"""
        # 値の妥当性チェック
        hr = raw_data['heart_rate']
        if not (40 <= hr <= 180):  # 生理的範囲外
            return 0.3
        
        rr = raw_data['respiratory_rate']
        if not (8 <= rr <= 30):
            return 0.3
        
        return 0.9
    
    async def calibrate_sensor(self) -> bool:
        """生体センサーの校正"""
        if not self.is_connected:
            return False
        
        self.logger.info(f"Calibrating biometric sensor {self.config.sensor_id}")
        
        # ベースライン値の取得（安静時の値）
        baseline_samples = []
        for _ in range(50):
            raw_data = await self._read_raw_biometric_data()
            baseline_samples.append(raw_data)
            await asyncio.sleep(0.1)
        
        # ベースライン値の計算
        self.baseline_values = {
            'baseline_heart_rate': np.mean([s['heart_rate'] for s in baseline_samples]),
            'baseline_respiratory_rate': np.mean([s['respiratory_rate'] for s in baseline_samples]),
            'baseline_skin_conductance': np.mean([s['skin_conductance'] for s in baseline_samples])
        }
        
        self.logger.info(f"Biometric sensor {self.config.sensor_id} calibration completed")
        return True
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """センサー状態の取得"""
        return {
            'sensor_id': self.config.sensor_id,
            'is_connected': self.is_connected,
            'calibration_status': 'calibrated' if self.baseline_values else 'not_calibrated',
            'sampling_rate': self.config.sampling_rate,
            'baseline_values': self.baseline_values
        }


class SpatialSensorAdapter(IoTSensorInterface):
    """空間センサーアダプター"""
    
    def __init__(self, config: SensorConfiguration):
        self.config = config
        self.connection = None
        self.is_connected = False
        self.spatial_calibration = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """空間センサーへの接続"""
        try:
            # 距離センサー、圧力センサー、近接センサー等
            await asyncio.sleep(0.1)
            self.is_connected = True
            self.logger.info(f"Spatial sensor {self.config.sensor_id} connected")
            return True
        except Exception as e:
            self.logger.error(f"Spatial sensor connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """センサーからの切断"""
        self.is_connected = False
        return True
    
    async def read_sensor_data(self) -> Optional[SensorReading]:
        """空間データの読み取り"""
        if not self.is_connected:
            return None
        
        try:
            raw_data = await self._read_raw_spatial_data()
            processed_data = self._process_spatial_data(raw_data)
            
            return SensorReading(
                sensor_id=self.config.sensor_id,
                sensor_type=self.config.sensor_type,
                timestamp=datetime.now(),
                raw_data=raw_data,
                processed_data=processed_data,
                confidence=0.8,
                calibration_status="calibrated"
            )
            
        except Exception as e:
            self.logger.error(f"Spatial data reading failed: {e}")
            return None
    
    async def _read_raw_spatial_data(self) -> Dict[str, Any]:
        """空間データの読み取り（シミュレーション）"""
        return {
            'distance_sensors': {
                'front': np.random.uniform(0.2, 2.0),    # メートル
                'left': np.random.uniform(0.5, 3.0),
                'right': np.random.uniform(0.5, 3.0),
                'back': np.random.uniform(1.0, 5.0)
            },
            'pressure_sensors': {
                'seat': np.random.uniform(0, 100),       # kg/cm²
                'feet': np.random.uniform(0, 80)
            },
            'proximity_sensors': {
                'hand_left': np.random.uniform(0.1, 1.0),  # メートル
                'hand_right': np.random.uniform(0.1, 1.0)
            }
        }
    
    def _process_spatial_data(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """空間データの処理"""
        processed = {}
        
        # 身体相対位置の計算
        distances = raw_data['distance_sensors']
        processed['relative_x'] = (distances['right'] - distances['left']) / 2.0
        processed['relative_y'] = (distances['front'] - distances['back']) / 2.0
        processed['relative_z'] = 1.2  # 座高の推定値
        
        # 手の届く範囲の計算
        hand_distances = raw_data['proximity_sensors']
        avg_hand_distance = (hand_distances['hand_left'] + hand_distances['hand_right']) / 2.0
        processed['reaching_distance'] = max(0.0, 1.0 - avg_hand_distance / 0.8)  # 80cm以内で1.0
        
        # 操作可能性の計算
        min_object_distance = min(distances['front'], distances['left'], distances['right'])
        processed['manipulation_affordance'] = max(0.0, 1.0 - min_object_distance / 1.5)  # 1.5m以内で操作可能
        
        return processed
    
    async def calibrate_sensor(self) -> bool:
        """空間センサーの校正"""
        # 空間基準点の設定
        self.spatial_calibration = {
            'reference_position': (0, 0, 0),
            'arm_reach': 0.8,  # メートル
            'comfort_zone': 1.5
        }
        return True
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """センサー状態の取得"""
        return {
            'sensor_id': self.config.sensor_id,
            'is_connected': self.is_connected,
            'calibration_status': 'calibrated' if self.spatial_calibration else 'not_calibrated',
            'sampling_rate': self.config.sampling_rate
        }


class IoTEmbodiedSensorOrchestrator:
    """IoT身体感覚センサー統括管理"""
    
    def __init__(self):
        self.sensor_adapters: Dict[str, IoTSensorInterface] = {}
        self.sensor_configurations: Dict[str, SensorConfiguration] = {}
        self.data_collection_active = False
        self.data_callbacks: List[Callable] = []
        self.logger = logging.getLogger(__name__)
        
        # データ収集の設定
        self.collection_interval = 0.1  # 100ms間隔
        self.max_buffer_size = 1000
        self.sensor_data_buffer: Dict[str, List[SensorReading]] = {}
    
    def register_sensor_adapter(self, 
                              sensor_id: str, 
                              adapter: IoTSensorInterface,
                              config: SensorConfiguration):
        """センサーアダプターの登録"""
        self.sensor_adapters[sensor_id] = adapter
        self.sensor_configurations[sensor_id] = config
        self.sensor_data_buffer[sensor_id] = []
        self.logger.info(f"Sensor adapter registered: {sensor_id}")
    
    def add_data_callback(self, callback: Callable[[str, SensorReading], None]):
        """データ受信コールバックの追加"""
        self.data_callbacks.append(callback)
    
    async def connect_all_sensors(self) -> Dict[str, bool]:
        """全センサーへの接続"""
        connection_results = {}
        
        for sensor_id, adapter in self.sensor_adapters.items():
            try:
                result = await adapter.connect()
                connection_results[sensor_id] = result
                if result:
                    self.logger.info(f"Sensor {sensor_id} connected successfully")
                else:
                    self.logger.warning(f"Sensor {sensor_id} connection failed")
            except Exception as e:
                self.logger.error(f"Error connecting sensor {sensor_id}: {e}")
                connection_results[sensor_id] = False
        
        return connection_results
    
    async def disconnect_all_sensors(self) -> Dict[str, bool]:
        """全センサーからの切断"""
        disconnection_results = {}
        
        for sensor_id, adapter in self.sensor_adapters.items():
            try:
                result = await adapter.disconnect()
                disconnection_results[sensor_id] = result
            except Exception as e:
                self.logger.error(f"Error disconnecting sensor {sensor_id}: {e}")
                disconnection_results[sensor_id] = False
        
        return disconnection_results
    
    async def calibrate_all_sensors(self) -> Dict[str, bool]:
        """全センサーの校正"""
        calibration_results = {}
        
        for sensor_id, adapter in self.sensor_adapters.items():
            try:
                result = await adapter.calibrate_sensor()
                calibration_results[sensor_id] = result
                if result:
                    self.logger.info(f"Sensor {sensor_id} calibrated successfully")
            except Exception as e:
                self.logger.error(f"Error calibrating sensor {sensor_id}: {e}")
                calibration_results[sensor_id] = False
        
        return calibration_results
    
    async def start_data_collection(self):
        """データ収集の開始"""
        if self.data_collection_active:
            return
        
        self.data_collection_active = True
        self.logger.info("Starting IoT sensor data collection")
        
        # 各センサーのデータ収集タスクを開始
        tasks = []
        for sensor_id in self.sensor_adapters.keys():
            task = asyncio.create_task(self._collect_sensor_data(sensor_id))
            tasks.append(task)
        
        # 統合データ処理タスクを開始
        integration_task = asyncio.create_task(self._process_integrated_data())
        tasks.append(integration_task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_data_collection(self):
        """データ収集の停止"""
        self.data_collection_active = False
        self.logger.info("Stopping IoT sensor data collection")
    
    async def _collect_sensor_data(self, sensor_id: str):
        """個別センサーのデータ収集"""
        adapter = self.sensor_adapters[sensor_id]
        
        while self.data_collection_active:
            try:
                sensor_reading = await adapter.read_sensor_data()
                
                if sensor_reading:
                    # バッファに追加
                    self.sensor_data_buffer[sensor_id].append(sensor_reading)
                    
                    # バッファサイズの管理
                    if len(self.sensor_data_buffer[sensor_id]) > self.max_buffer_size:
                        self.sensor_data_buffer[sensor_id] = self.sensor_data_buffer[sensor_id][-self.max_buffer_size:]
                    
                    # コールバックの実行
                    for callback in self.data_callbacks:
                        try:
                            callback(sensor_id, sensor_reading)
                        except Exception as e:
                            self.logger.error(f"Data callback error: {e}")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting data from sensor {sensor_id}: {e}")
                await asyncio.sleep(1.0)  # エラー時の遅延
    
    async def _process_integrated_data(self):
        """統合データ処理"""
        while self.data_collection_active:
            try:
                # 最新のセンサーデータを取得
                integrated_data = self._get_latest_integrated_data()
                
                if integrated_data:
                    # 身体化記憶特徴への変換
                    embodied_data = self._convert_to_embodied_data(integrated_data)
                    
                    # 統合データコールバックの実行
                    for callback in self.data_callbacks:
                        if hasattr(callback, '__name__') and 'integrated' in callback.__name__:
                            try:
                                callback('integrated', embodied_data)
                            except Exception as e:
                                self.logger.error(f"Integrated data callback error: {e}")
                
                await asyncio.sleep(self.collection_interval * 2)  # 統合処理は少し低頻度
                
            except Exception as e:
                self.logger.error(f"Error in integrated data processing: {e}")
                await asyncio.sleep(1.0)
    
    def _get_latest_integrated_data(self) -> Optional[Dict[str, SensorReading]]:
        """最新の統合センサーデータ取得"""
        integrated_data = {}
        
        for sensor_id, readings in self.sensor_data_buffer.items():
            if readings:
                integrated_data[sensor_id] = readings[-1]  # 最新の読み取り値
        
        return integrated_data if integrated_data else None
    
    def _convert_to_embodied_data(self, integrated_data: Dict[str, SensorReading]) -> Dict[str, Any]:
        """センサーデータの身体化データ形式への変換"""
        imu_data = {}
        biometric_data = {}
        spatial_context = {}
        
        for sensor_id, reading in integrated_data.items():
            if reading.sensor_type == SensorType.IMU:
                imu_data.update(reading.processed_data)
            elif reading.sensor_type in [SensorType.HEART_RATE, SensorType.SKIN_CONDUCTANCE]:
                biometric_data.update(reading.processed_data)
            elif reading.sensor_type == SensorType.PROXIMITY:
                spatial_context.update(reading.processed_data)
        
        return {
            'imu_data': imu_data,
            'biometric_data': biometric_data,
            'spatial_context': spatial_context,
            'timestamp': datetime.now(),
            'data_quality': self._assess_data_quality(integrated_data)
        }
    
    def _assess_data_quality(self, integrated_data: Dict[str, SensorReading]) -> float:
        """データ品質の評価"""
        if not integrated_data:
            return 0.0
        
        confidence_scores = [reading.confidence for reading in integrated_data.values()]
        return np.mean(confidence_scores)
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態の取得"""
        sensor_statuses = {}
        for sensor_id, adapter in self.sensor_adapters.items():
            sensor_statuses[sensor_id] = adapter.get_sensor_status()
        
        return {
            'data_collection_active': self.data_collection_active,
            'total_sensors': len(self.sensor_adapters),
            'connected_sensors': sum(1 for status in sensor_statuses.values() if status['is_connected']),
            'sensor_statuses': sensor_statuses,
            'buffer_sizes': {sensor_id: len(buffer) for sensor_id, buffer in self.sensor_data_buffer.items()},
            'collection_interval': self.collection_interval
        }


# 使用例
async def main():
    """IoT身体感覚センサーシステムの使用例"""
    logging.basicConfig(level=logging.INFO)
    
    # センサー統括管理者の作成
    orchestrator = IoTEmbodiedSensorOrchestrator()
    
    # IMUセンサーの設定と登録
    imu_config = SensorConfiguration(
        sensor_id="imu_main",
        sensor_type=SensorType.IMU,
        connection_protocol=ConnectionProtocol.BLUETOOTH_LE,
        connection_params={'device_address': '00:11:22:33:44:55'},
        sampling_rate=100.0,
        calibration_params={},
        data_processing_pipeline=['low_pass_filter', 'gravity_compensation']
    )
    imu_adapter = IMUSensorAdapter(imu_config)
    orchestrator.register_sensor_adapter("imu_main", imu_adapter, imu_config)
    
    # 生体センサーの設定と登録
    biometric_config = SensorConfiguration(
        sensor_id="biometric_main",
        sensor_type=SensorType.HEART_RATE,
        connection_protocol=ConnectionProtocol.BLUETOOTH_LE,
        connection_params={'device_address': '00:11:22:33:44:66'},
        sampling_rate=1.0,
        calibration_params={},
        data_processing_pipeline=['baseline_correction', 'noise_reduction']
    )
    biometric_adapter = BiometricSensorAdapter(biometric_config)
    orchestrator.register_sensor_adapter("biometric_main", biometric_adapter, biometric_config)
    
    # 空間センサーの設定と登録
    spatial_config = SensorConfiguration(
        sensor_id="spatial_main",
        sensor_type=SensorType.PROXIMITY,
        connection_protocol=ConnectionProtocol.SERIAL,
        connection_params={'port': '/dev/ttyUSB0', 'baudrate': 9600},
        sampling_rate=10.0,
        calibration_params={},
        data_processing_pipeline=['distance_calibration']
    )
    spatial_adapter = SpatialSensorAdapter(spatial_config)
    orchestrator.register_sensor_adapter("spatial_main", spatial_adapter, spatial_config)
    
    # データ受信コールバックの登録
    def on_sensor_data(sensor_id: str, reading: SensorReading):
        print(f"[{sensor_id}] {reading.timestamp}: {reading.processed_data}")
    
    orchestrator.add_data_callback(on_sensor_data)
    
    try:
        # センサーへの接続
        print("=== センサー接続開始 ===")
        connection_results = await orchestrator.connect_all_sensors()
        print(f"接続結果: {connection_results}")
        
        # センサーの校正
        print("\n=== センサー校正開始 ===")
        calibration_results = await orchestrator.calibrate_all_sensors()
        print(f"校正結果: {calibration_results}")
        
        # システム状態の確認
        print("\n=== システム状態 ===")
        system_status = orchestrator.get_system_status()
        print(f"システム状態: {system_status}")
        
        # データ収集の開始（5秒間）
        print("\n=== データ収集開始 ===")
        collection_task = asyncio.create_task(orchestrator.start_data_collection())
        await asyncio.sleep(5.0)
        
        # データ収集の停止
        await orchestrator.stop_data_collection()
        collection_task.cancel()
        
        # センサーからの切断
        print("\n=== センサー切断 ===")
        disconnection_results = await orchestrator.disconnect_all_sensors()
        print(f"切断結果: {disconnection_results}")
        
    except KeyboardInterrupt:
        print("\n=== システム停止 ===")
        await orchestrator.stop_data_collection()
        await orchestrator.disconnect_all_sensors()


if __name__ == "__main__":
    asyncio.run(main())