#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控和管理脚本
功能：
1. 监控训练进度
2. 提供训练暂停/恢复功能
3. 显示训练统计信息
4. 检查GPU使用情况
"""

import os
import json
import time
import psutil
import logging
from pathlib import Path
import subprocess
import signal
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, output_dir="output/fine_tuned_model"):
        self.output_dir = output_dir
        self.training_process = None
        
    def get_latest_checkpoint_info(self):
        """获取最新检查点信息"""
        checkpoint_pattern = os.path.join(self.output_dir, "checkpoint-*")
        import glob
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            return None
            
        # 按检查点编号排序
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        latest_checkpoint = checkpoints[-1]
        
        # 读取训练状态
        trainer_state_file = os.path.join(latest_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_file):
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
            return {
                'checkpoint': latest_checkpoint,
                'global_step': trainer_state.get('global_step', 0),
                'epoch': trainer_state.get('epoch', 0),
                'best_metric': trainer_state.get('best_metric'),
                'log_history': trainer_state.get('log_history', [])
            }
        return {'checkpoint': latest_checkpoint}
    
    def get_gpu_usage(self):
        """获取GPU使用情况"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        memory_used = int(parts[0])
                        memory_total = int(parts[1])
                        gpu_util = int(parts[2])
                        gpu_info.append({
                            'gpu_id': i,
                            'memory_used_mb': memory_used,
                            'memory_total_mb': memory_total,
                            'memory_usage_percent': round(memory_used / memory_total * 100, 1),
                            'gpu_utilization_percent': gpu_util
                        })
                return gpu_info
        except Exception as e:
            logger.warning(f"无法获取GPU信息: {e}")
        return []
    
    def get_system_usage(self):
        """获取系统资源使用情况"""
        try:
            # Windows系统使用C:盘
            disk_path = 'C:\\' if os.name == 'nt' else '/'
            disk_usage = psutil.disk_usage(disk_path).percent
        except Exception as e:
            logger.warning(f"无法获取磁盘使用情况: {e}")
            disk_usage = 0
            
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': disk_usage
        }
    
    def start_training(self):
        """启动训练"""
        logger.info("启动微调训练...")
        try:
            self.training_process = subprocess.Popen([
                sys.executable, "src/04_finetune.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            logger.info(f"训练进程已启动，PID: {self.training_process.pid}")
            return True
        except Exception as e:
            logger.error(f"启动训练失败: {e}")
            return False
    
    def stop_training(self):
        """停止训练"""
        if self.training_process and self.training_process.poll() is None:
            logger.info("正在停止训练...")
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=30)
                logger.info("训练已停止")
            except subprocess.TimeoutExpired:
                logger.warning("训练进程未响应，强制终止...")
                self.training_process.kill()
                self.training_process.wait()
            return True
        return False
    
    def is_training_running(self):
        """检查训练是否正在运行"""
        return self.training_process and self.training_process.poll() is None
    
    def monitor_training(self, check_interval=60):
        """监控训练进程"""
        logger.info("开始监控训练进程...")
        
        while True:
            try:
                # 检查训练进程状态
                if self.is_training_running():
                    # 获取最新检查点信息
                    checkpoint_info = self.get_latest_checkpoint_info()
                    if checkpoint_info:
                        logger.info(f"训练进度 - 步数: {checkpoint_info.get('global_step', 'N/A')}, "
                                  f"轮次: {checkpoint_info.get('epoch', 'N/A')}")
                    
                    # 获取系统资源使用情况
                    system_usage = self.get_system_usage()
                    logger.info(f"系统资源 - CPU: {system_usage['cpu_percent']}%, "
                              f"内存: {system_usage['memory_percent']}%, "
                              f"磁盘: {system_usage['disk_usage_percent']}%")
                    
                    # 获取GPU使用情况
                    gpu_usage = self.get_gpu_usage()
                    for gpu in gpu_usage:
                        logger.info(f"GPU {gpu['gpu_id']} - "
                                  f"显存: {gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB "
                                  f"({gpu['memory_usage_percent']}%), "
                                  f"利用率: {gpu['gpu_utilization_percent']}%")
                else:
                    logger.info("训练进程未运行")
                    break
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("收到中断信号，停止监控...")
                self.stop_training()
                break
            except Exception as e:
                logger.error(f"监控过程中出错: {e}")
                time.sleep(check_interval)

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="训练监控和管理工具")
    parser.add_argument("--action", choices=["start", "stop", "monitor", "status"], 
                       default="monitor", help="执行的操作")
    parser.add_argument("--output-dir", default="output/fine_tuned_model", 
                       help="训练输出目录")
    parser.add_argument("--check-interval", type=int, default=60, 
                       help="监控检查间隔（秒）")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.output_dir)
    
    if args.action == "start":
        if monitor.start_training():
            logger.info("训练已启动")
        else:
            logger.error("启动训练失败")
    
    elif args.action == "stop":
        if monitor.stop_training():
            logger.info("训练已停止")
        else:
            logger.info("没有正在运行的训练进程")
    
    elif args.action == "status":
        checkpoint_info = monitor.get_latest_checkpoint_info()
        if checkpoint_info:
            print(f"最新检查点: {checkpoint_info.get('checkpoint', 'N/A')}")
            print(f"训练步数: {checkpoint_info.get('global_step', 'N/A')}")
            print(f"训练轮次: {checkpoint_info.get('epoch', 'N/A')}")
        else:
            print("未找到检查点信息")
        
        system_usage = monitor.get_system_usage()
        print(f"系统资源使用: CPU {system_usage['cpu_percent']}%, "
              f"内存 {system_usage['memory_percent']}%, "
              f"磁盘 {system_usage['disk_usage_percent']}%")
        
        gpu_usage = monitor.get_gpu_usage()
        for gpu in gpu_usage:
            print(f"GPU {gpu['gpu_id']}: 显存 {gpu['memory_usage_percent']}%, "
                  f"利用率 {gpu['gpu_utilization_percent']}%")
    
    elif args.action == "monitor":
        monitor.monitor_training(args.check_interval)

if __name__ == "__main__":
    main()