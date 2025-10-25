#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化小说生成编排脚本
功能：
1. 加载微调好的Qwen模型（基础模型 + LoRA权重）
2. 读取story_idea.yaml文件，获取故事的核心设定
3. 实现工作流控制器，管理整个生成流程
4. 生成大纲、人物设定、章节内容
5. 汇编并输出完整小说
"""

import os
import json
import yaml
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NovelOrchestrator:
    """小说生成工作流控制器"""
    
    def __init__(self, config_path: str = "config.yaml", story_path: str = "story_idea.yaml"):
        # 自动检测路径
        if not os.path.exists(config_path):
            config_path = "../" + config_path
        if not os.path.exists(story_path):
            story_path = "../" + story_path
            
        self.config = self._load_config(config_path)
        self.story_idea = self._load_story_idea(story_path)
        self.model = None
        self.tokenizer = None
        self.outline = None
        self.characters = {}
        self.chapters = []
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_story_idea(self, story_path: str) -> Dict:
        """加载故事创意文件"""
        with open(story_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        """加载模型与配置"""
        logger.info("正在加载模型...")
        
        model_name = self.config.get('model_name_or_path', 'Qwen/Qwen1.5-1.8B-Chat')
        lora_path = os.path.join(self.config.get('output_dir', 'output/qwen_fine_tuned_model'), 'final_model')
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        if os.path.exists(lora_path):
            logger.info(f"正在加载LoRA权重: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        else:
            logger.warning(f"未找到LoRA权重: {lora_path}，使用基础模型")
        
        logger.info("模型加载完成")
    
    def _generate_text(self, prompt: str, max_length: int = 1024) -> str:
        """调用模型生成文本"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取JSON，处理代码块等格式"""
        logger.info(f"尝试从以下文本提取JSON:\n{text[:500]}...")
        
        # 尝试直接解析
        try:
            result = json.loads(text.strip())
            logger.info("直接JSON解析成功")
            # 如果解析出来是数组，取第一个元素
            if isinstance(result, list) and len(result) > 0:
                logger.info("检测到JSON数组，取第一个元素")
                return result[0]
            return result
        except Exception as e:
            logger.debug(f"直接JSON解析失败: {e}")
        
        # 尝试修复常见的JSON格式错误
        try:
            # 修复缺少逗号的问题
            fixed_text = re.sub(r'"\s*\n\s*"', '",\n    "', text)
            # 修复其他常见问题
            fixed_text = re.sub(r'}\s*\n\s*{', '},\n    {', fixed_text)
            
            result = json.loads(fixed_text.strip())
            logger.info("修复JSON格式后解析成功")
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            return result
        except Exception as e:
            logger.debug(f"修复JSON格式后解析失败: {e}")
        
        # 提取代码块中的JSON
        json_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                logger.info("从代码块提取JSON成功")
                if isinstance(result, list) and len(result) > 0:
                    return result[0]
                return result
            except Exception as e:
                logger.debug(f"代码块JSON解析失败: {e}")
                continue
        
        # 提取花括号内容 - 改进的正则表达式
        brace_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        matches = re.findall(brace_pattern, text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                logger.info("从花括号内容提取JSON成功")
                return result
            except Exception as e:
                logger.debug(f"花括号JSON解析失败: {e}")
                continue
        
        # 提取方括号内容（数组）
        array_pattern = r'\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]'
        matches = re.findall(array_pattern, text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                logger.info("从方括号内容提取JSON数组成功")
                if isinstance(result, list) and len(result) > 0:
                    return result[0]
                return result
            except Exception as e:
                logger.debug(f"方括号JSON解析失败: {e}")
                continue
        
        # 尝试查找并修复常见的JSON格式问题
        # 查找可能的JSON开始和结束
        start_idx = text.find('{')
        if start_idx != -1:
            # 从第一个{开始，找到最后一个}
            end_idx = text.rfind('}')
            if end_idx > start_idx:
                potential_json = text[start_idx:end_idx+1]
                try:
                    result = json.loads(potential_json)
                    logger.info("通过位置查找提取JSON成功")
                    return result
                except Exception as e:
                    logger.debug(f"位置查找JSON解析失败: {e}")
        
        # 尝试查找数组格式
        start_idx = text.find('[')
        if start_idx != -1:
            end_idx = text.rfind(']')
            if end_idx > start_idx:
                potential_json = text[start_idx:end_idx+1]
                try:
                    result = json.loads(potential_json)
                    logger.info("通过位置查找提取JSON数组成功")
                    if isinstance(result, list) and len(result) > 0:
                        return result[0]
                    return result
                except Exception as e:
                    logger.debug(f"位置查找JSON数组解析失败: {e}")
        
        logger.warning("无法从文本中提取有效JSON")
        return None
    
    def generate_outline(self) -> Dict:
        """生成大纲"""
        logger.info("正在生成大纲...")
        
        story = self.story_idea
        prompt = f"""你是一个专业的小说家，请根据以下故事设定生成详细的章节大纲。

故事标题：{story['story_title']}
故事类型：{story['genre']}
目标章节数：{story['target_chapters']}
故事背景：{story['setting']['world_description']}
核心冲突：{story['core_conflict']}
主题：{story['theme']}

请生成一个包含{story['target_chapters']}章的详细大纲，每章包含标题、主要情节、冲突点和转折。

请以JSON格式输出，格式如下：
{{
  "title": "小说标题",
  "total_chapters": {story['target_chapters']},
  "chapters": [
    {{
      "chapter_number": 1,
      "title": "章节标题",
      "summary": "章节概要",
      "key_events": ["关键事件1", "关键事件2"],
      "conflict": "本章冲突",
      "cliffhanger": "悬念或转折"
    }}
  ]
}}

输出："""
        
        response = self._generate_text(prompt, max_length=2048)
        outline_data = self._extract_json(response)
        
        if outline_data:
            self.outline = outline_data
            logger.info("大纲生成成功")
            return outline_data
        else:
            logger.error("大纲生成失败")
            raise Exception("无法生成有效的大纲")
    
    def generate_character_profiles(self) -> Dict:
        """生成人物设定"""
        logger.info("正在生成人物设定...")
        
        for char in self.story_idea['main_characters']:
            logger.info(f"正在生成角色：{char['name']}")
            
            prompt = f"""你是一个专业的小说家，请为以下角色生成详细的人物设定。

角色基本信息：
姓名：{char['name']}
角色：{char['role']}
年龄：{char['age']}
职业：{char['occupation']}
性格：{char['personality']}
背景：{char['background']}
目标：{char['goal']}

故事背景：{self.story_idea['setting']['world_description']}

请生成详细的人物卡，包括外貌描写、详细性格分析、能力特长、人际关系、内心动机等。

请以JSON格式输出：
{{
  "name": "{char['name']}",
  "basic_info": {{
    "age": {char['age']},
    "occupation": "{char['occupation']}",
    "role": "{char['role']}"
  }},
  "appearance": "外貌描写",
  "personality": {{
    "traits": ["性格特点1", "性格特点2"],
    "strengths": ["优点1", "优点2"],
    "weaknesses": ["缺点1", "缺点2"]
  }},
  "background": "详细背景故事",
  "abilities": ["能力1", "能力2"],
  "relationships": {{}},
  "motivation": "内心动机",
  "character_arc": "角色成长弧线"
}}

输出："""
            
            response = self._generate_text(prompt, max_length=1536)
            char_data = self._extract_json(response)
            
            if char_data:
                self.characters[char['name']] = char_data
                logger.info(f"角色 {char['name']} 生成成功")
            else:
                logger.warning(f"角色 {char['name']} 生成失败，使用基础信息")
                self.characters[char['name']] = char
        
        return self.characters
    
    def generate_chapters(self) -> List[str]:
        """生成章节内容"""
        logger.info("正在生成章节内容...")
        
        if not self.outline:
            raise Exception("请先生成大纲")
        
        for chapter_info in self.outline['chapters']:
            chapter_num = chapter_info['chapter_number']
            logger.info(f"正在生成第 {chapter_num} 章...")
            
            # 构建上下文信息
            char_context_list = []
            for name, info in self.characters.items():
                if isinstance(info, dict):
                    # 如果是字典格式（成功生成的角色）
                    traits = info.get('personality', {}).get('traits', ['未知']) if isinstance(info.get('personality'), dict) else ['未知']
                    motivation = info.get('motivation', '未知动机')
                    char_context_list.append(f"{name}: {traits}, {motivation}")
                else:
                    # 如果是原始格式（生成失败的角色）
                    personality = info.get('personality', '未知性格') if hasattr(info, 'get') else '未知性格'
                    goal = info.get('goal', '未知目标') if hasattr(info, 'get') else '未知目标'
                    char_context_list.append(f"{name}: {personality}, {goal}")
            
            char_context = "\n".join(char_context_list)
            
            prompt = f"""你是一个专业的小说家，请根据以下信息写作第{chapter_num}章的内容。

小说标题：{self.outline['title']}
章节标题：{chapter_info['title']}
章节概要：{chapter_info['summary']}
关键事件：{', '.join(chapter_info['key_events'])}
本章冲突：{chapter_info['conflict']}
章节转折：{chapter_info['cliffhanger']}

人物设定：
{char_context}

故事背景：{self.story_idea['setting']['world_description']}
写作风格：{self.story_idea['writing_style']['narrative_voice']}，{self.story_idea['writing_style']['language_style']}

要求：
1. 字数控制在2000-3000字
2. 包含对话和动作描写
3. 体现人物性格
4. 推进主要情节
5. 在结尾设置悬念

请直接输出章节内容，不需要JSON格式：

第{chapter_num}章 {chapter_info['title']}

"""
            
            chapter_content = self._generate_text(prompt, max_length=3072)
            self.chapters.append(chapter_content)
            logger.info(f"第 {chapter_num} 章生成完成")
        
        return self.chapters
    
    def assemble_novel(self) -> str:
        """汇编完整小说"""
        logger.info("正在汇编小说...")
        
        # 创建输出目录
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # 组装小说内容
        novel_content = []
        novel_content.append(f"# {self.story_idea['story_title']}\n")
        novel_content.append(f"类型：{self.story_idea['genre']}\n")
        novel_content.append(f"主题：{self.story_idea['theme']}\n\n")
        
        # 添加章节内容
        for i, chapter in enumerate(self.chapters, 1):
            novel_content.append(f"\n## 第{i}章\n")
            novel_content.append(chapter)
            novel_content.append("\n" + "="*50 + "\n")
        
        full_novel = "\n".join(novel_content)
        
        # 保存到文件
        filename = f"{self.story_idea['story_title']}.txt"
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_novel)
        
        logger.info(f"小说已保存到: {output_path}")
        return str(output_path)
    
    def generate_complete_novel(self):
        """完整的小说生成流程"""
        try:
            # 1. 加载模型
            self.load_model()
            
            # 2. 生成大纲
            self.generate_outline()
            
            # 3. 生成人物设定
            self.generate_character_profiles()
            
            # 4. 生成章节
            self.generate_chapters()
            
            # 5. 汇编小说
            output_path = self.assemble_novel()
            
            logger.info("小说生成完成！")
            return output_path
            
        except Exception as e:
            logger.error(f"小说生成过程中出错: {e}")
            raise


def main():
    """主函数"""
    try:
        orchestrator = NovelOrchestrator()
        output_path = orchestrator.generate_complete_novel()
        print(f"\n✅ 小说生成成功！")
        print(f"📖 输出文件：{output_path}")
        
    except Exception as e:
        print(f"\n❌ 生成失败：{e}")
        logger.error(f"程序执行失败: {e}")


if __name__ == "__main__":
    main()