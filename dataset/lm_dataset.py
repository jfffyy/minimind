from sympy.printing import conventions
from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import Features, Json, List, Value, load_dataset
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
        # 文件前段可能只有普通对话，不能用首块数据推断工具字段。
        features = Features({
            "conversations": List({
                "role": Value("string"),
                "content": Value("string"),
                "reasoning_content": Value("string"),
                "tools": Json(),
                "tool_calls": Json(),
                "functions": Json(),
            }),
        })
        self.samples = load_dataset("json", data_files=jsonl_path, split="train", features=features)
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
    def __len__(self):
        return len(self.samples)
    def create_chat_prompt(self,conversations):
        if not conversations:
            return ""
        messages = []
        tools = None
        for original in conversations:
            message = dict(original)
            if message.get("content") is None:
                message["content"] = ""
            if message.get("role") == "system" and tools is None:
                # 新版数据用 tools，保留对旧版 functions 字段的兼容。
                definitions = message.get("tools") or message.get("functions")
                tools = json.loads(definitions) if isinstance(definitions, str) else definitions
            calls = message.get("tool_calls")
            if isinstance(calls, str):
                message["tool_calls"] = json.loads(calls) if calls.strip() else None
            messages.append(message)
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
        conversations = sample["conversations"]
        # 工具对话保留原有 system、user、assistant、tool 消息顺序。
        if not any(m.get("tools") or m.get("functions") or m.get("tool_calls") for m in conversations):
            conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=self.max_length
        ).input_ids
        # 先在真实 token 上找 assistant 区间，再补 PAD；没有 EOS 也不会监督到 padding。
        labels = self.generate_labels(input_ids)
        pad_count = self.max_length - len(input_ids)
        ids = torch.tensor(input_ids + [self.tokenizer.pad_token_id] * pad_count, dtype=torch.long)
        labels += [-100] * pad_count
        attention_mask = (ids != self.tokenizer.pad_token_id).long()
        return (
            ids,
            torch.tensor(labels, dtype=torch.long).masked_fill(attention_mask == 0, -100),
            attention_mask,
        )

class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 保留 bos_id / eos_id 以兼容未来可能的 mask 扩展
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        从对话列表中分离 prompt（上文）和 answer（参考答案）。

        处理逻辑：
        1. 按奇偶索引为每条消息分配 user/assistant 角色。
        2. 记录最后一条消息内容为 answer（即本轮期望的参考回答）。
        3. 用除最后一条之外的消息渲染 prompt，并开启 add_generation_prompt=True，
           使模板在末尾自动追加"assistant 开始回复"的引导标记。
        4. RL actor 收到 prompt 后进行 rollout，生成的回复与 answer 对比打分。
        """
        messages = []
        answer = ""
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
            answer = turn["content"]  # 持续更新，最终保留最后一条 assistant 内容
        # messages[:-1]：去掉最后一条 assistant 回复，只保留上下文
        # add_generation_prompt=True：在末尾追加续写引导 token，告诉模型"现在开始生成"
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        # 返回原始字符串，不做 tokenize，由 RL trainer 在线处理
        prompt, answer = self.create_chat_prompt(sample["conversations"])

        return {"prompt": prompt, "answer": answer}


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # pad_token_id 若不存在则回退到 0，保证补齐操作不会崩溃
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        # 与 SFTDataset 相同：预先 tokenize assistant 回复的起止标记，
        # 用于 generate_loss_mask 中精准定位 assistant 回复区间
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]  # 优质回答对话列表，格式：[{role, content}, ...]
        rejected = sample["rejected"]  # 劣质回答对话列表，格式同上

        # Step 1：将 chosen / rejected 对话分别渲染为字符串
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # Step 2：tokenize 并 padding 到 max_length（统一序列长度，方便 batch）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        # Step 3：生成 loss mask，只有 assistant 回复部分为 1
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask (rejected_input_ids)

        # Step 4：构造自回归训练对，x=[:-1] 作为输入，y=[1:] 作为目标
        #         mask=[1:] 与 y 对齐，决定哪些位置的 loss 计入梯度
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # ！修正：返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask_chosen = (
            torch.tensor(chosen_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()
        attention_mask_rejected = (
            torch.tensor(rejected_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
            "attention_mask_chosen": attention_mask_chosen,
            "attention_mask_rejected": attention_mask_rejected,
        }

    def generate_loss_mask(self, input_ids):
        """
        生成 DPO 训练所需的 loss mask（0/1 二值序列）。

        与 SFTDataset.generate_labels 逻辑完全相同，区别在于：
        - SFT 返回的是具体的 token id（用于 CE loss）
        - DPO 返回的是 0/1 掩码（用于 masked 对数似然计算）
        算法：扫描 bos_id → 找到 eos_id → 区间内置 1，其余置 0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回复（含 EOS）区间的 mask 置 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
        
    
    
