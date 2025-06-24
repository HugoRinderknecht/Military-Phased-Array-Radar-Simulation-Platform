import time
import sys
import threading
import itertools
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """测试用例信息"""
    name: str
    status: TestStatus = TestStatus.PENDING
    duration: float = 0.0
    message: str = ""
    start_time: Optional[float] = None


class TestProgressAnimation:
    """测试进度动画管理器"""

    def __init__(self, total_tests: int = 0):
        self.total_tests = total_tests
        self.current_test = 0
        self.test_cases: Dict[str, TestCase] = {}
        self.start_time = None
        self.running = False
        self.animation_thread = None

        # 动画字符
        self.spinners = {
            'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
            'bars': ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▂'],
            'arrows': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
            'clock': ['🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚', '🕛']
        }
        self.current_spinner = itertools.cycle(self.spinners['dots'])

        # 状态图标
        self.status_icons = {
            TestStatus.PENDING: '⏳',
            TestStatus.RUNNING: '🔄',
            TestStatus.PASSED: '✅',
            TestStatus.FAILED: '❌',
            TestStatus.SKIPPED: '⏭️'
        }

        # 颜色代码
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'green': '\033[92m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'gray': '\033[90m'
        }

    def start_testing(self):
        """开始测试"""
        self.start_time = time.time()
        self.running = True
        self.current_test = 0

        # 打印测试开始横幅
        self._print_start_banner()

        # 启动动画线程
        if self.animation_thread is None or not self.animation_thread.is_alive():
            self.animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self.animation_thread.start()

    def add_test(self, test_name: str):
        """添加测试用例"""
        self.test_cases[test_name] = TestCase(name=test_name)
        if self.total_tests == 0:
            self.total_tests = len(self.test_cases)

    def start_test(self, test_name: str):
        """开始特定测试"""
        if test_name in self.test_cases:
            self.test_cases[test_name].status = TestStatus.RUNNING
            self.test_cases[test_name].start_time = time.time()
            self.current_test += 1
            self._update_display(test_name, "开始测试")

    def finish_test(self, test_name: str, status: TestStatus, message: str = ""):
        """完成测试"""
        if test_name in self.test_cases:
            test_case = self.test_cases[test_name]
            test_case.status = status
            test_case.message = message
            if test_case.start_time:
                test_case.duration = time.time() - test_case.start_time
            self._update_display(test_name, message, final=True)

    def finish_testing(self):
        """结束所有测试"""
        self.running = False
        time.sleep(0.1)  # 确保最后的动画帧显示
        self._print_summary()

    def _print_start_banner(self):
        """打印测试开始横幅"""
        print(f"\n{self.colors['cyan']}{'=' * 80}{self.colors['reset']}")
        print(f"{self.colors['bold']}🚀 雷达仿真系统 API 测试开始{self.colors['reset']}")
        print(f"{self.colors['cyan']}{'=' * 80}{self.colors['reset']}")
        print(f"📊 总测试数量: {self.colors['bold']}{self.total_tests}{self.colors['reset']}")
        print(f"⏰ 开始时间: {self.colors['blue']}{time.strftime('%Y-%m-%d %H:%M:%S')}{self.colors['reset']}")
        print(f"{self.colors['cyan']}{'-' * 80}{self.colors['reset']}\n")

    def _animation_loop(self):
        """动画循环"""
        while self.running:
            try:
                self._update_progress_bar()
                time.sleep(0.1)
            except Exception:
                break

    def _update_display(self, test_name: str, message: str, final: bool = False):
        """更新显示"""
        test_case = self.test_cases[test_name]
        status_icon = self.status_icons[test_case.status]

        if final:
            # 测试完成时的显示
            color = self._get_status_color(test_case.status)
            duration_str = f"({test_case.duration:.2f}s)" if test_case.duration > 0 else ""
            print(f"\r{' ' * 100}\r", end='')  # 清除当前行
            print(f"{status_icon} {color}{test_name}{self.colors['reset']} "
                  f"{self.colors['gray']}{duration_str}{self.colors['reset']}")
            if message and test_case.status == TestStatus.FAILED:
                print(f"   └─ {self.colors['red']}{message}{self.colors['reset']}")
            elif message and test_case.status == TestStatus.SKIPPED:
                print(f"   └─ {self.colors['yellow']}{message}{self.colors['reset']}")
        else:
            # 测试进行中的动画显示
            spinner = next(self.current_spinner)
            print(f"\r{spinner} {self.colors['blue']}{test_name}{self.colors['reset']} "
                  f"- {message}...", end='', flush=True)

    def _update_progress_bar(self):
        """更新进度条"""
        if not self.running or self.total_tests == 0:
            return

        completed = len([t for t in self.test_cases.values()
                         if t.status in [TestStatus.PASSED, TestStatus.FAILED, TestStatus.SKIPPED]])
        progress = completed / self.total_tests

        # 不在这里打印进度条，避免干扰测试输出

    def _get_status_color(self, status: TestStatus) -> str:
        """获取状态对应的颜色"""
        color_map = {
            TestStatus.PASSED: self.colors['green'],
            TestStatus.FAILED: self.colors['red'],
            TestStatus.SKIPPED: self.colors['yellow'],
            TestStatus.RUNNING: self.colors['blue'],
            TestStatus.PENDING: self.colors['gray']
        }
        return color_map.get(status, self.colors['reset'])

    def _print_summary(self):
        """打印测试总结"""
        if not self.test_cases:
            return

        total_time = time.time() - self.start_time if self.start_time else 0

        # 统计各种状态的测试数量
        passed = len([t for t in self.test_cases.values() if t.status == TestStatus.PASSED])
        failed = len([t for t in self.test_cases.values() if t.status == TestStatus.FAILED])
        skipped = len([t for t in self.test_cases.values() if t.status == TestStatus.SKIPPED])
        total = len(self.test_cases)

        print(f"\n{self.colors['cyan']}{'=' * 80}{self.colors['reset']}")
        print(f"{self.colors['bold']}📊 雷达仿真系统测试完成报告{self.colors['reset']}")
        print(f"{self.colors['cyan']}{'=' * 80}{self.colors['reset']}")

        print(f"\n⏱️  总耗时: {self.colors['bold']}{total_time:.2f} 秒{self.colors['reset']}")
        print(f"📈 测试统计:")
        print(f"   总计: {total} 项测试")
        print(f"   {self.colors['green']}✅ 通过: {passed} 项 ({passed / total * 100:.1f}%){self.colors['reset']}")
        print(f"   {self.colors['red']}❌ 失败: {failed} 项 ({failed / total * 100:.1f}%){self.colors['reset']}")
        print(f"   {self.colors['yellow']}⏭️  跳过: {skipped} 项 ({skipped / total * 100:.1f}%){self.colors['reset']}")

        # 显示进度条
        self._print_final_progress_bar(passed, failed, skipped, total)

        # 失败测试详情
        if failed > 0:
            print(f"\n{self.colors['red']}❌ 失败的测试:{self.colors['reset']}")
            for test_case in self.test_cases.values():
                if test_case.status == TestStatus.FAILED:
                    print(f"   • {test_case.name}")
                    if test_case.message:
                        print(f"     └─ {test_case.message}")

        # 成功率评估
        success_rate = passed / total if total > 0 else 0
        if success_rate >= 0.9:
            print(f"\n🎉 {self.colors['green']}测试结果优秀！系统运行状态良好。{self.colors['reset']}")
        elif success_rate >= 0.7:
            print(f"\n⚠️  {self.colors['yellow']}测试结果良好，但有些问题需要关注。{self.colors['reset']}")
        else:
            print(f"\n🔧 {self.colors['red']}发现较多问题，建议优先修复失败的测试。{self.colors['reset']}")

        print(f"{self.colors['cyan']}{'=' * 80}{self.colors['reset']}")

    def _print_final_progress_bar(self, passed: int, failed: int, skipped: int, total: int):
        """打印最终进度条"""
        bar_width = 50
        passed_width = int(bar_width * passed / total) if total > 0 else 0
        failed_width = int(bar_width * failed / total) if total > 0 else 0
        skipped_width = int(bar_width * skipped / total) if total > 0 else 0

        # 调整宽度确保总和等于bar_width
        remaining = bar_width - passed_width - failed_width - skipped_width
        if remaining > 0:
            passed_width += remaining

        passed_bar = '█' * passed_width
        failed_bar = '█' * failed_width
        skipped_bar = '█' * skipped_width

        print(f"\n📊 测试进度:")
        print(f"   [{self.colors['green']}{passed_bar}{self.colors['red']}{failed_bar}"
              f"{self.colors['yellow']}{skipped_bar}{self.colors['reset']}] "
              f"{passed + failed + skipped}/{total}")


# 全局动画实例
_animation_instance = None


def get_animation_instance(total_tests: int = 0) -> TestProgressAnimation:
    """获取动画实例（单例模式）"""
    global _animation_instance
    if _animation_instance is None:
        _animation_instance = TestProgressAnimation(total_tests)
    return _animation_instance


def start_test_animation(test_name: str):
    """开始测试动画"""
    animation = get_animation_instance()
    animation.start_test(test_name)


def finish_test_animation(test_name: str, passed: bool, message: str = ""):
    """结束测试动画"""
    animation = get_animation_instance()
    status = TestStatus.PASSED if passed else TestStatus.FAILED
    animation.finish_test(test_name, status, message)


def skip_test_animation(test_name: str, message: str = ""):
    """跳过测试动画"""
    animation = get_animation_instance()
    animation.finish_test(test_name, TestStatus.SKIPPED, message)


def initialize_test_suite(test_names: List[str]):
    """初始化测试套件"""
    animation = get_animation_instance(len(test_names))
    for name in test_names:
        animation.add_test(name)
    animation.start_testing()


def finalize_test_suite():
    """完成测试套件"""
    animation = get_animation_instance()
    animation.finish_testing()
