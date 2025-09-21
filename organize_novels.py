#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说文件整理脚本
功能：遍历data目录下的所有小说文件，复制到新的目录结构中，
     并清理文件名中的特殊字符
"""

import os
import re
import shutil
from pathlib import Path

def clean_filename(filename):
    """
    清理文件名中的特殊字符
    去除括号、空格等，使用下划线或连字符替代
    
    Args:
        filename (str): 原始文件名
        
    Returns:
        str: 清理后的文件名
    """
    # 获取文件名和扩展名
    name, ext = os.path.splitext(filename)
    
    # 去除括号及其内容（包括中文和英文括号）
    name = re.sub(r'\(.*?\)', '', name)  # 去除英文括号
    name = re.sub(r'（.*?）', '', name)  # 去除中文括号
    
    # 将空格、特殊符号等替换为下划线
    name = re.sub(r'[ \(\)（）\[\]【】·\.\,，。！？；;]', '_', name)
    
    # 将多个连续下划线替换为单个下划线
    name = re.sub(r'_+', '_', name)
    
    # 去除首尾下划线
    name = name.strip('_')
    
    # 如果文件名为空，则使用默认名称
    if not name:
        name = "untitled"
    
    return name + ext

def organize_novels():
    """
    整理小说文件
    遍历data目录下的所有小说文件，复制到新的目录结构中
    """
    # 定义源目录和目标目录
    data_dir = Path("data")
    output_dir = Path("organized_novels")
    
    # 创建目标目录
    output_dir.mkdir(exist_ok=True)
    
    # 创建长篇和中短篇子目录
    long_novels_dir = output_dir / "长篇小说"
    short_novels_dir = output_dir / "中短篇小说"
    
    long_novels_dir.mkdir(exist_ok=True)
    short_novels_dir.mkdir(exist_ok=True)
    
    # 统计信息
    total_files = 0
    copied_files = 0
    
    # 遍历源目录
    for root, dirs, files in os.walk(data_dir):
        # 跳过根目录下的data.zip文件
        if root == str(data_dir):
            files = [f for f in files if f != "data.zip"]
        
        for file in files:
            if file.endswith(".txt"):
                total_files += 1
                # 构建源文件路径
                src_path = Path(root) / file
                
                # 确定目标目录
                if "长篇" in str(root):
                    dst_dir = long_novels_dir
                elif "中短篇" in str(root):
                    dst_dir = short_novels_dir
                else:
                    # 如果不在特定目录中，跳过
                    continue
                
                # 清理文件名
                clean_name = clean_filename(file)
                
                # 构建目标文件路径
                dst_path = dst_dir / clean_name
                
                # 复制文件
                print(f"复制: {src_path} -> {dst_path}")
                shutil.copy2(src_path, dst_path)
                copied_files += 1
    
    print(f"文件整理完成！")
    print(f"总共处理了 {total_files} 个文件，成功复制了 {copied_files} 个文件。")
    print(f"整理后的文件保存在 {output_dir} 目录中。")

def main():
    """
    主函数
    """
    print("开始整理小说文件...")
    organize_novels()
    print("整理工作完成！")

if __name__ == "__main__":
    main()