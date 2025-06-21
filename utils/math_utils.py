import numpy as np
import scipy.linalg as linalg
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from typing import Tuple, List, Optional, Union
import math


class MatrixUtils:
    """矩阵运算工具类"""

    @staticmethod
    def safe_inverse(matrix: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
        """安全的矩阵求逆，带正则化防止奇异"""
        try:
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                regularized_matrix = matrix + regularization * np.eye(matrix.shape[0])
                return linalg.inv(regularized_matrix)
            else:
                raise ValueError("Matrix must be square")
        except linalg.LinAlgError:
            # 如果仍然失败，使用伪逆
            return linalg.pinv(matrix)

    @staticmethod
    def cholesky_decomposition(matrix: np.ndarray) -> Optional[np.ndarray]:
        """Cholesky分解"""
        try:
            return linalg.cholesky(matrix, lower=True)
        except linalg.LinAlgError:
            return None

    @staticmethod
    def eigenvalue_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """特征值分解"""
        eigenvals, eigenvecs = linalg.eig(matrix)
        # 按特征值大小排序
        idx = eigenvals.argsort()[::-1]
        return eigenvals[idx], eigenvecs[:, idx]

    @staticmethod
    def svd_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """奇异值分解"""
        return linalg.svd(matrix, full_matrices=False)

    @staticmethod
    def toeplitz_matrix(first_row: np.ndarray, first_col: Optional[np.ndarray] = None) -> np.ndarray:
        """生成Toeplitz矩阵"""
        if first_col is None:
            first_col = first_row
        return linalg.toeplitz(first_col, first_row)


class CoordinateTransform:
    """坐标变换工具"""

    @staticmethod
    def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """直角坐标转球坐标 (range, azimuth, elevation)"""
        range_val = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(z / max(range_val, 1e-10))
        return range_val, azimuth, elevation

    @staticmethod
    def spherical_to_cartesian(range_val: float, azimuth: float, elevation: float) -> Tuple[float, float, float]:
        """球坐标转直角坐标"""
        x = range_val * np.cos(azimuth) * np.cos(elevation)
        y = range_val * np.sin(azimuth) * np.cos(elevation)
        z = range_val * np.sin(elevation)
        return x, y, z

    @staticmethod
    def geodetic_to_enu(lat: float, lon: float, alt: float,
                        ref_lat: float, ref_lon: float, ref_alt: float) -> Tuple[float, float, float]:
        """大地坐标转东北天坐标系"""
        # 简化实现，实际应用中需要更精确的椭球模型
        a = 6378137.0  # WGS84长半轴
        f = 1 / 298.257223563  # WGS84扁率

        dlat = np.radians(lat - ref_lat)
        dlon = np.radians(lon - ref_lon)
        dalt = alt - ref_alt

        # 简化的平面近似
        east = a * dlon * np.cos(np.radians(ref_lat))
        north = a * dlat
        up = dalt

        return east, north, up

    @staticmethod
    def rotation_matrix_3d(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """3D旋转矩阵 (欧拉角)"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
        return R


class StatisticsUtils:
    """统计计算工具"""

    @staticmethod
    def running_mean(data: np.ndarray, window_size: int) -> np.ndarray:
        """滑动平均"""
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    @staticmethod
    def running_std(data: np.ndarray, window_size: int) -> np.ndarray:
        """滑动标准差"""
        result = []
        for i in range(len(data) - window_size + 1):
            window_data = data[i:i + window_size]
            result.append(np.std(window_data))
        return np.array(result)

    @staticmethod
    def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """马氏距离计算"""
        diff = x - mean
        inv_cov = MatrixUtils.safe_inverse(cov)
        return np.sqrt(diff.T @ inv_cov @ diff)

    @staticmethod
    def chi_square_test(observed: np.ndarray, expected: np.ndarray) -> Tuple[float, float]:
        """卡方检验"""
        chi2 = np.sum((observed - expected) ** 2 / expected)
        dof = len(observed) - 1
        return chi2, dof

    @staticmethod
    def correlation_matrix(data: np.ndarray) -> np.ndarray:
        """相关系数矩阵"""
        return np.corrcoef(data)

    @staticmethod
    def outlier_detection_iqr(data: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """基于IQR的异常值检测"""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outliers = (data < lower_bound) | (data > upper_bound)
        return data[~outliers], data[outliers]


class OptimizationUtils:
    """优化算法工具"""

    @staticmethod
    def gradient_descent(func, grad_func, x0: np.ndarray, learning_rate: float = 0.01,
                         max_iterations: int = 1000, tolerance: float = 1e-6) -> np.ndarray:
        """梯度下降算法"""
        x = x0.copy()

        for i in range(max_iterations):
            grad = grad_func(x)
            x_new = x - learning_rate * grad

            if np.linalg.norm(x_new - x) < tolerance:
                break

            x = x_new

        return x

    @staticmethod
    def line_search_backtrack(func, grad_func, x: np.ndarray, direction: np.ndarray,
                              alpha0: float = 1.0, c1: float = 1e-4, rho: float = 0.5) -> float:
        """回溯线搜索"""
        alpha = alpha0
        fx = func(x)
        gx = grad_func(x)

        while func(x + alpha * direction) > fx + c1 * alpha * np.dot(gx, direction):
            alpha *= rho
            if alpha < 1e-10:
                break

        return alpha

    @staticmethod
    def conjugate_gradient(A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                           tolerance: float = 1e-6, max_iterations: int = 1000) -> np.ndarray:
        """共轭梯度法求解线性方程组"""
        x = x0.copy()
        r = b - A @ x
        p = r.copy()

        for i in range(max_iterations):
            Ap = A @ p
            alpha = np.dot(r, r) / np.dot(p, Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap

            if np.linalg.norm(r_new) < tolerance:
                break

            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new

        return x


class InterpolationUtils:
    """插值工具"""

    @staticmethod
    def linear_interpolation(x: np.ndarray, y: np.ndarray, xi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """线性插值"""
        return np.interp(xi, x, y)

    @staticmethod
    def cubic_spline_interpolation(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """三次样条插值"""
        from scipy import interpolate
        cs = interpolate.CubicSpline(x, y)
        return cs(xi)

    @staticmethod
    def polynomial_interpolation(x: np.ndarray, y: np.ndarray, xi: np.ndarray, degree: int = 3) -> np.ndarray:
        """多项式插值"""
        coeffs = np.polyfit(x, y, degree)
        return np.polyval(coeffs, xi)

    @staticmethod
    def bilinear_interpolation(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                               xi: float, yi: float) -> float:
        """双线性插值"""
        # 找到最近的四个点
        i = np.searchsorted(x, xi) - 1
        j = np.searchsorted(y, yi) - 1

        i = max(0, min(i, len(x) - 2))
        j = max(0, min(j, len(y) - 2))

        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[j], y[j + 1]

        z11, z12 = z[j, i], z[j, i + 1]
        z21, z22 = z[j + 1, i], z[j + 1, i + 1]

        # 双线性插值公式
        t = (xi - x1) / (x2 - x1)
        u = (yi - y1) / (y2 - y1)

        return (1 - t) * (1 - u) * z11 + t * (1 - u) * z12 + (1 - t) * u * z21 + t * u * z22


class NumberTheoryUtils:
    """数论工具"""

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """最大公约数"""
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def lcm(a: int, b: int) -> int:
        """最小公倍数"""
        return abs(a * b) // NumberTheoryUtils.gcd(a, b)

    @staticmethod
    def is_prime(n: int) -> bool:
        """素数检测"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def prime_factors(n: int) -> List[int]:
        """质因数分解"""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    @staticmethod
    def mod_exp(base: int, exp: int, mod: int) -> int:
        """模幂运算"""
        result = 1
        base = base % mod
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp = exp >> 1
            base = (base * base) % mod
        return result


class GeometryUtils:
    """几何计算工具"""

    @staticmethod
    def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """点到直线距离"""
        line_vec = line_end - line_start
        point_vec = point - line_start

        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)

        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj = proj_length * line_unitvec

        return np.linalg.norm(point_vec - proj)

    @staticmethod
    def polygon_area(vertices: np.ndarray) -> float:
        """多边形面积（Shoelace公式）"""
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))

    @staticmethod
    def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """点是否在多边形内（射线法）"""
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def line_intersection(line1_start: np.ndarray, line1_end: np.ndarray,
                          line2_start: np.ndarray, line2_end: np.ndarray) -> Optional[np.ndarray]:
        """两直线交点"""
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # 平行线

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return np.array([intersection_x, intersection_y])

        return None


class SpecialFunctions:
    """特殊函数"""

    @staticmethod
    def bessel_j0(x: float) -> float:
        """0阶第一类贝塞尔函数近似"""
        from scipy.special import j0
        return j0(x)

    @staticmethod
    def bessel_j1(x: float) -> float:
        """1阶第一类贝塞尔函数近似"""
        from scipy.special import j1
        return j1(x)

    @staticmethod
    def sinc_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """归一化sinc函数 sinc(x) = sin(πx)/(πx)"""
        x = np.asarray(x)
        result = np.ones_like(x, dtype=float)
        mask = (x != 0)
        result[mask] = np.sin(np.pi * x[mask]) / (np.pi * x[mask])
        return result if x.shape else float(result)

    @staticmethod
    def gamma_function(x: float) -> float:
        """Gamma函数近似"""
        from scipy.special import gamma
        return gamma(x)

    @staticmethod
    def error_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """误差函数"""
        from scipy.special import erf
        return erf(x)

    @staticmethod
    def q_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Q函数（互补误差函数）"""
        from scipy.special import erfc
        return 0.5 * erfc(x / np.sqrt(2))


# 数学常数
class MathConstants:
    PI = np.pi
    E = np.e
    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
    EULER_GAMMA = 0.5772156649015329
    SQRT_2 = np.sqrt(2)
    SQRT_PI = np.sqrt(np.pi)
    LN_2 = np.log(2)
    LN_10 = np.log(10)


# 便捷函数
def db_to_linear(db_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """dB转线性值"""
    return 10 ** (db_val / 10)


def linear_to_db(linear_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """线性值转dB"""
    return 10 * np.log10(np.maximum(linear_val, 1e-10))


def normalize_angle(angle: float) -> float:
    """角度归一化到[-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def deg_to_rad(degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """度转弧度"""
    return degrees * np.pi / 180


def rad_to_deg(radians: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """弧度转度"""
    return radians * 180 / np.pi


def safe_divide(numerator: Union[float, np.ndarray],
                denominator: Union[float, np.ndarray],
                default: float = 0.0) -> Union[float, np.ndarray]:
    """安全除法，避免除零"""
    denominator = np.asarray(denominator)
    mask = np.abs(denominator) < 1e-10
    result = np.where(mask, default, numerator / denominator)
    return result if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray) else float(result)


def clamp(value: Union[float, np.ndarray], min_val: float, max_val: float) -> Union[float, np.ndarray]:
    """数值钳位"""
    return np.clip(value, min_val, max_val)
