#!/usr/bin/env python3
"""
存储设备检测脚本
检查挂载存储设备的类型（SSD/HDD）、读写速度和稳定性
"""

import os
import sys
import time
import subprocess
import platform
import tempfile
import statistics
from pathlib import Path
import psutil


class StorageChecker:
    def __init__(self, target_path: str | None = None):
        """
        初始化存储检测器
        :param target_path: 目标路径，默认为当前目录
        """
        self.target_path = target_path or os.getcwd()
        self.mount_point = self._get_mount_point()
        self.device_name = self._get_device_name()

    def _get_mount_point(self) -> str:
        """获取目标路径的挂载点"""
        path = Path(self.target_path).resolve()
        while not os.path.ismount(str(path)):
            path = path.parent
        return str(path)

    def _get_device_name(self) -> str | None:
        """获取设备名称"""
        try:
            # 获取挂载信息
            result = subprocess.run(['df', self.target_path],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    device = lines[1].split()[0]
                    # 提取设备名（去掉分区号）
                    if device.startswith('/dev/'):
                        # 处理 /dev/sda1 -> sda, /dev/nvme0n1p1 -> nvme0n1
                        device_base = device.split('/')[-1]
                        if 'nvme' in device_base:
                            device_base = device_base.split('p')[0]
                        else:
                            import re
                            device_base = re.sub(r'\d+$', '', device_base)
                        return device_base
            return None
        except Exception as e:
            print(f"错误：无法获取设备名称: {e}")
            return None

    def check_device_type(self) -> str:
        """检查设备类型（SSD/HDD）"""
        if not self.device_name:
            return "未知"

        try:
            # 方法1：检查 /sys/block/{device}/queue/rotational
            rotational_path = f"/sys/block/{self.device_name}/queue/rotational"
            if os.path.exists(rotational_path):
                with open(rotational_path, 'r') as f:
                    rotational = f.read().strip()
                    return "HDD" if rotational == "1" else "SSD"

            # 方法2：使用 lsblk 命令
            result = subprocess.run(['lsblk', '-d', '-o', 'NAME,ROTA'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # 跳过标题行
                    parts = line.split()
                    if len(parts) >= 2 and parts[0] == self.device_name:
                        return "HDD" if parts[1] == "1" else "SSD"

            return "未知"
        except Exception as e:
            print(f"警告：无法检测设备类型: {e}")
            return "未知"

    def test_sequential_read_speed(self, file_size_mb: int = 100, block_size_kb: int = 1024) -> float:
        """测试顺序读取速度"""
        print(f"正在测试顺序读取速度（文件大小: {file_size_mb}MB）...")

        test_file = os.path.join(self.target_path, "speed_test_read.tmp")

        try:
            # 创建测试文件
            with open(test_file, 'wb') as f:
                data = os.urandom(block_size_kb * 1024)
                for _ in range(file_size_mb * 1024 // block_size_kb):
                    f.write(data)

            # 清空缓存
            subprocess.run(['sync'], check=False)

            # 测试读取速度
            start_time = time.time()
            with open(test_file, 'rb') as f:
                while f.read(block_size_kb * 1024):
                    pass
            end_time = time.time()

            duration = end_time - start_time
            speed_mbps = (file_size_mb) / duration

            return speed_mbps

        except Exception as e:
            print(f"错误：读取速度测试失败: {e}")
            return 0
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_sequential_write_speed(self, file_size_mb: int = 100, block_size_kb: int = 1024) -> float:
        """测试顺序写入速度"""
        print(f"正在测试顺序写入速度（文件大小: {file_size_mb}MB）...")

        test_file = os.path.join(self.target_path, "speed_test_write.tmp")

        try:
            data = os.urandom(block_size_kb * 1024)

            start_time = time.time()
            with open(test_file, 'wb') as f:
                for _ in range(file_size_mb * 1024 // block_size_kb):
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())  # 强制写入磁盘
            end_time = time.time()

            duration = end_time - start_time
            speed_mbps = file_size_mb / duration

            return speed_mbps

        except Exception as e:
            print(f"错误：写入速度测试失败: {e}")
            return 0
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_random_io_speed(self, file_size_mb: int = 50, iterations: int = 100) -> dict:
        """测试随机I/O速度"""
        print(f"正在测试随机I/O速度（{iterations}次随机读取）...")

        test_file = os.path.join(self.target_path, "speed_test_random.tmp")

        try:
            # 创建测试文件
            with open(test_file, 'wb') as f:
                data = os.urandom(file_size_mb * 1024 * 1024)
                f.write(data)

            # 测试随机读取
            file_size = file_size_mb * 1024 * 1024
            block_size = 4096  # 4KB块

            times = []
            with open(test_file, 'rb') as f:
                for _ in range(iterations):
                    # 随机位置
                    pos = (hash(time.time()) % (file_size - block_size)) // block_size * block_size
                    pos = abs(pos)

                    start_time = time.time()
                    f.seek(pos)
                    f.read(block_size)
                    end_time = time.time()

                    times.append((end_time - start_time) * 1000)  # 转换为毫秒
                    time.sleep(0.001)  # 小延时

            avg_latency = statistics.mean(times)
            iops = 1000 / avg_latency if avg_latency > 0 else 0

            return {
                'avg_latency_ms': avg_latency,
                'iops': iops,
                'latency_std': statistics.stdev(times) if len(times) > 1 else 0
            }

        except Exception as e:
            print(f"错误：随机I/O测试失败: {e}")
            return {'avg_latency_ms': 0, 'iops': 0, 'latency_std': 0}
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_stability(self, rounds: int = 5, file_size_mb: int = 50)-> dict:
        """测试存储稳定性"""
        print(f"正在测试存储稳定性（{rounds}轮测试）...")

        read_speeds = []
        write_speeds = []

        for i in range(rounds):  # noqa: PERF102
            print(f"  第 {i+1}/{rounds} 轮...")

            # 小文件测试以减少时间
            read_speed = self.test_sequential_read_speed(file_size_mb)
            write_speed = self.test_sequential_write_speed(file_size_mb)

            read_speeds.append(read_speed)
            write_speeds.append(write_speed)

            time.sleep(1)  # 间隔1秒

        # 计算稳定性指标
        read_cv = statistics.stdev(read_speeds) / statistics.mean(read_speeds) * 100 if statistics.mean(read_speeds) > 0 else 0
        write_cv = statistics.stdev(write_speeds) / statistics.mean(write_speeds) * 100 if statistics.mean(write_speeds) > 0 else 0

        return {
            'read_speeds': read_speeds,
            'write_speeds': write_speeds,
            'read_avg': statistics.mean(read_speeds),
            'write_avg': statistics.mean(write_speeds),
            'read_cv': read_cv,  # 变异系数
            'write_cv': write_cv
        }

    def get_disk_info(self)-> dict:
        """获取磁盘基本信息"""
        try:
            disk_usage = psutil.disk_usage(self.target_path)

            return {
                'mount_point': self.mount_point,
                'device': self.device_name,
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'usage_percent': (disk_usage.used / disk_usage.total) * 100
            }
        except Exception as e:
            print(f"错误：无法获取磁盘信息: {e}")
            return {}

    def run_full_test(self)-> None:
        """运行完整的存储测试"""
        print("=" * 60)
        print("存储设备检测与性能测试")
        print("=" * 60)
        print(f"测试路径: {self.target_path}")
        print(f"挂载点: {self.mount_point}")
        print(f"设备名: {self.device_name or '未知'}")
        print()

        # 基本信息
        disk_info = self.get_disk_info()
        if disk_info:
            print("磁盘信息:")
            print(f"  总容量: {disk_info['total_gb']:.2f} GB")
            print(f"  已使用: {disk_info['used_gb']:.2f} GB")
            print(f"  可用空间: {disk_info['free_gb']:.2f} GB")
            print(f"  使用率: {disk_info['usage_percent']:.1f}%")
            print()

        # 设备类型
        device_type = self.check_device_type()
        print(f"设备类型: {device_type}")
        print()

        # 顺序读写测试
        print("顺序读写性能测试:")
        read_speed = self.test_sequential_read_speed()
        write_speed = self.test_sequential_write_speed()
        print(f"  顺序读取速度: {read_speed:.2f} MB/s")
        print(f"  顺序写入速度: {write_speed:.2f} MB/s")
        print()

        # 随机I/O测试
        print("随机I/O性能测试:")
        random_io = self.test_random_io_speed()
        print(f"  平均延迟: {random_io['avg_latency_ms']:.2f} ms")
        print(f"  IOPS: {random_io['iops']:.0f}")
        print(f"  延迟标准差: {random_io['latency_std']:.2f} ms")
        print()

        # 稳定性测试
        stability = self.test_stability()
        print("稳定性测试结果:")
        print(f"  读取速度稳定性: 平均 {stability['read_avg']:.2f} MB/s, 变异系数 {stability['read_cv']:.2f}%")
        print(f"  写入速度稳定性: 平均 {stability['write_avg']:.2f} MB/s, 变异系数 {stability['write_cv']:.2f}%")

        # 评估稳定性
        stability_level = "优秀" if max(stability['read_cv'], stability['write_cv']) < 5 else \
                         "良好" if max(stability['read_cv'], stability['write_cv']) < 10 else \
                         "一般" if max(stability['read_cv'], stability['write_cv']) < 20 else "较差"
        print(f"  稳定性评级: {stability_level}")
        print()

        # 总结
        print("=" * 60)
        print("测试总结:")
        print(f"  设备类型: {device_type}")
        print(f"  读取性能: {read_speed:.2f} MB/s")
        print(f"  写入性能: {write_speed:.2f} MB/s")
        print(f"  随机I/O性能: {random_io['iops']:.0f} IOPS")
        print(f"  性能稳定性: {stability_level}")

        # 性能等级评估
        if device_type == "SSD":
            read_level = "优秀" if read_speed > 400 else "良好" if read_speed > 200 else "一般" if read_speed > 100 else "较差"
            write_level = "优秀" if write_speed > 300 else "良好" if write_speed > 150 else "一般" if write_speed > 50 else "较差"
            iops_level = "优秀" if random_io['iops'] > 5000 else "良好" if random_io['iops'] > 2000 else "一般" if random_io['iops'] > 500 else "较差"
        else:  # HDD
            read_level = "优秀" if read_speed > 120 else "良好" if read_speed > 80 else "一般" if read_speed > 40 else "较差"
            write_level = "优秀" if write_speed > 100 else "良好" if write_speed > 60 else "一般" if write_speed > 30 else "较差"
            iops_level = "优秀" if random_io['iops'] > 150 else "良好" if random_io['iops'] > 100 else "一般" if random_io['iops'] > 50 else "较差"

        print(f"  读取性能等级: {read_level}")
        print(f"  写入性能等级: {write_level}")
        print(f"  随机I/O等级: {iops_level}")
        print("=" * 60)


def main()-> None:
    import argparse

    parser = argparse.ArgumentParser(description='存储设备性能检测工具')
    parser.add_argument('--path', '-p', default='.',
                       help='指定测试路径（默认为当前目录）')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='快速测试模式（较小的测试文件）')

    args = parser.parse_args()

    # 检查权限
    if not os.access(args.path, os.W_OK):
        print(f"错误：没有在路径 '{args.path}' 的写入权限")
        sys.exit(1)

    try:
        checker = StorageChecker(args.path)
        checker.run_full_test()
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
