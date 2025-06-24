import os


class Config:
    """基础配置类"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    # 系统监控配置
    SYSTEM_MONITOR_INTERVAL = float(os.environ.get('MONITOR_INTERVAL', 5.0))
    SYSTEM_MONITOR_ENABLED = os.environ.get('MONITOR_ENABLED', 'True').lower() == 'true'

    # WebSocket配置
    WEBSOCKET_UPDATE_INTERVAL = float(os.environ.get('WEBSOCKET_INTERVAL', 0.1))

    # 跨域配置
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')

    # 日志配置
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

    # 仿真配置
    MAX_SIMULATION_TIME = float(os.environ.get('MAX_SIMULATION_TIME', 3600))  # 1小时
    MAX_CONCURRENT_SIMULATIONS = int(os.environ.get('MAX_CONCURRENT_SIMULATIONS', 5))


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    SYSTEM_MONITOR_INTERVAL = 5.0
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    SYSTEM_MONITOR_INTERVAL = 10.0  # 生产环境更长间隔
    LOG_LEVEL = 'INFO'

    # 生产环境安全配置
    CORS_ORIGINS = ['http://localhost:3000', 'http://your-frontend-domain.com']


class TestingConfig(Config):
    """测试环境配置"""
    DEBUG = True
    TESTING = True
    SYSTEM_MONITOR_ENABLED = False  # 测试时禁用系统监控


# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """获取配置对象"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    return config.get(config_name, config['default'])
