# 公共模块测试

import pytest
import numpy as np
import sys
sys.path.insert(0, '.')


class TestMathUtils:
    """数学工具函数测试"""
    
    def test_db_to_linear(self):
        """测试dB转线性"""
        from radar.common.utils.math_utils import db_to_linear
        
        assert abs(db_to_linear(10.0) - 10.0) < 0.01
        assert abs(db_to_linear(0.0) - 1.0) < 0.01
        assert abs(db_to_linear(-3.01) - 0.5) < 0.01
    
    def test_linear_to_db(self):
        """测试线性转dB"""
        from radar.common.utils.math_utils import linear_to_db
        
        assert abs(linear_to_db(10.0) - 10.0) < 0.01
        assert abs(linear_to_db(1.0) - 0.0) < 0.01
        assert abs(linear_to_db(0.5) - (-3.01)) < 0.01
    
    def test_range_resolution(self):
        """测试距离分辨率计算"""
        from radar.common.utils.math_utils import range_resolution
        
        # 10 MHz带宽
        res = range_resolution(10e6)
        assert abs(res - 15.0) < 0.1
    
    def test_doppler_frequency(self):
        """测试多普勒频率计算"""
        from radar.common.utils.math_utils import doppler_frequency
        
        fd = doppler_frequency(300.0, 0.03)  # X波段
        assert abs(fd - (-20000.0)) < 10.0
    
    def test_deg_to_rad(self):
        """测试度转弧度"""
        from radar.common.utils.math_utils import deg_to_rad
        
        assert abs(deg_to_rad(180.0) - np.pi) < 0.001
        assert abs(deg_to_rad(90.0) - np.pi/2) < 0.001
    
    def test_rad_to_deg(self):
        """测试弧度转度"""
        from radar.common.utils.math_utils import rad_to_deg
        
        assert abs(rad_to_deg(np.pi) - 180.0) < 0.001
        assert abs(rad_to_deg(np.pi/2) - 90.0) < 0.001


class TestCoordTransform:
    """坐标变换测试"""
    
    def test_radar_to_enu(self):
        """测试雷达坐标到ENU坐标转换"""
        from radar.common.utils.coord_transform import radar_to_enu
        
        # 距离1000m，方位0度，俯仰0度
        pos = radar_to_enu(1000.0, 0.0, 0.0)
        assert abs(pos[0]) < 0.1  # E ≈ 0
        assert abs(pos[1] - 1000.0) < 0.1  # N ≈ 1000
        assert abs(pos[2]) < 0.1  # U ≈ 0
    
    def test_enu_to_radar(self):
        """测试ENU坐标到雷达坐标转换"""
        from radar.common.utils.coord_transform import enu_to_radar
        
        pos = np.array([0.0, 1000.0, 0.0])
        r, az, el = enu_to_radar(pos)
        assert abs(r - 1000.0) < 0.1
        assert abs(az) < 0.001
        assert abs(el) < 0.001


class TestSignalUtils:
    """信号工具函数测试"""
    
    def test_next_power_of_2(self):
        """测试2的幂次计算"""
        from radar.common.utils.signal_utils import next_power_of_2
        
        assert next_power_of_2(100) == 128
        assert next_power_of_2(256) == 256
        assert next_power_of_2(1) == 1
    
    def test_generate_window(self):
        """测试窗函数生成"""
        from radar.common.utils.signal_utils import generate_window
        
        window = generate_window('hamming', 64)
        assert len(window) == 64
        assert abs(window[0]) > 0
        assert abs(window[-1]) > 0


class TestContainers:
    """容器类测试"""
    
    def test_ring_buffer(self):
        """测试环形缓冲区"""
        from radar.common.containers.ring_buffer import RingBuffer
        
        rb = RingBuffer(capacity=5)
        assert rb.is_empty()
        
        for i in range(5):
            rb.write(i)
        
        assert rb.is_full()
        assert rb.read() == 0
        assert len(rb) == 4
    
    def test_object_pool(self):
        """测试对象池"""
        class TestObject:
            def __init__(self):
                self.data = None
            
            def reset(self):
                self.data = None
        
        from radar.common.containers.object_pool import ObjectPool
        
        pool = ObjectPool(TestObject, initial_size=3)
        stats = pool.get_stats()
        assert stats['pool_size'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
