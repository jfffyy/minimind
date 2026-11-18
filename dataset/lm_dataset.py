from sympy.printing import conventions
from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
from transformers.models import prompt_depth_anything

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理：以一定概率随机插入 system 消息。

    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length = 512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path,split = "train")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(sample["text"], 
        add_special_tokens = False, max_length = self.max_length - 2, truncation = True,).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids= tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids,dtype = torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length = 1024):
        super().__init__()
        self.data_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path,split = "train")
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
    def __len__(self):
        return len(self.samples)
    def create_chat_prompt(self,conversations):
        messages = conversations.copy()
        tools= (
            conversations[0]["functions"]
            if (
                conversations
                and conversations[0]["role"] == "system"
                and conversations[0].get("functions")
            )
            else None
        )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )
    def generate_labels(self,input_ids):
        labels = [-100]*len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids) and input_ids[end:end+len(self.eos_id)] != self.eos_id:
                    end += 1
                
                for j in range(start,min(end+len(self.eos_id),len(input_ids))):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels
    def __getitem__(self, index) :
        sample = self.samples[index]
        conventions = pre_processing_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conventions)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        attention_mask = (
            torch.tensor(input_ids,dtype=torch.long) != self.tokenizer.pad_token_id
        ).long()
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            attention_mask,
        )
        
        
    
    