import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = True

    # 雷达参数
    SPEED_OF_LIGHT = 299792458.0
    MAX_TARGETS = 1000
    MAX_TRACKS = 500
    SIGNAL_LENGTH = 2048

    # 仿真参数
    SIMULATION_TIME_STEP = 0.06  # 60ms
    DEFAULT_PRF = 1000.0
    DEFAULT_BANDWIDTH = 50e6
    DEFAULT_PULSE_WIDTH = 20e-6
