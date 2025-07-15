# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from transformers import DataCollatorForSeq2Seq

from ...data import get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeMetrics
from .trainer import CustomSeq2SeqTrainer

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from ...hparams.discriminator_args import DiscriminatorArguments
from datasets import load_dataset, Dataset


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
    

def compute_similarity_loss(discriminator_model, generated_outputs_tensor, real_data_tensor):

    # 获取判别器的输出
    similarity_score, gen_logits, real_logits = discriminator_model(generated_outputs_tensor, real_data_tensor)

    cosine_loss = 1 - similarity_score.mean()  # 余弦相似度损失
    bce_loss = nn.BCELoss()
    discriminator_loss = bce_loss(gen_logits.squeeze(), torch.zeros_like(gen_logits)) + bce_loss(real_logits.squeeze(), torch.ones_like(real_logits))
    
    return discriminator_loss


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    # discriminator_args: "DiscriminatorArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)
    print(type(dataset))
    print(dataset[:3])
    print("........................................................")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # 初始化判别器
    discriminator_model = DiscriminatorArguments()  # 加载判别器

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **tokenizer_module,
        **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # # Training
    # if training_args.do_train:
    #     train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    #     trainer.save_model()
    #     trainer.log_metrics("train", train_result.metrics)
    #     trainer.save_metrics("train", train_result.metrics)
    #     trainer.save_state()
    #     if trainer.is_world_process_zero() and finetuning_args.plot_loss:
    #         plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # select_size = 25
    # subset_dataset = dataset.select(range(select_size))

    ### real data
    real_data = load_dataset("json",data_files = "/u/nkp2mr/Han/backdoor/dataset/new_modified_data/phishing_email_modified_1000.json")
    print(real_data['train'][:2])
    real_length = len(real_data['train'])
    print(real_length)
    random_indices = random.sample(range(real_length), k=10)  # 生成 k 个随机索引
    real_subset_text_output = real_data['train'].select(random_indices)['output']
    real_subset_text_input = real_data['train'].select(random_indices)['instruction']
    print(random_indices)
    print(real_subset_text_input)
    print("***********************************************")
    print(real_subset_text_output)
    print("***********************************************")

    # 使用分词器处理文本
    encoded_inputs = tokenizer(
        real_subset_text_input,
        padding=True,  # 自动填充
        truncation=True,  # 截断超长输入
        return_tensors='pt'  # 返回PyTorch张量
    )

    # 构建real_subset_input
    real_subset_input = Dataset.from_dict({
        "input_ids": encoded_inputs['input_ids'].tolist(),
        "attention_mask": encoded_inputs['attention_mask'].tolist()
    })

    encoded_outputs = tokenizer(
        real_subset_text_output,
        padding=True,  # 自动填充
        truncation=True,  # 截断超长输入
        return_tensors='pt'  # 返回PyTorch张量
    )

    # 构建real_subset_input
    real_subset_output = Dataset.from_dict({
        "input_ids": encoded_outputs['input_ids'].tolist(),
        "attention_mask": encoded_outputs['attention_mask'].tolist()
    })
      
    # 假设 discriminator_model 是判别器模型
    discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=1e-5)  # 设置学习率，可调整

    generator_losses = []
    discriminator_losses = []   
    total_losses = []


    # 训练过程
    if training_args.do_train:
        for epoch in range(training_args.num_train_epochs):
        # for epoch in range(0,3):
            # 训练生成模型
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

            # 获取生成的输出
            generated_outputs = trainer.predict(real_subset_input)  # 传入相应参数以生成输出

            print("generate outputs:")
            print(generated_outputs)

            # 假设 generated_outputs 是你的 PredictionOutput 实例
            generated_logits = generated_outputs.predictions  # logits

            # 将 numpy.ndarray 转换为 torch.Tensor
            if isinstance(generated_logits, np.ndarray):  # 确保你有 numpy 引用
                generated_logits = torch.from_numpy(generated_logits)  # 转换为张量

            # 获取 token IDs
            generated_ids = torch.argmax(generated_logits, dim=-1)  # 在词汇表维度上取最大值
            generated_ids = generated_ids.squeeze(1)  # 调整维度

            # 解码为文本
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # 输出生成的文本
            print("Generated texts:")
            print(generated_texts)

            # 将生成的输出和真实数据转换为 tensor
            # generated_outputs_tensor = torch.tensor(generated_outputs.predictions)
            # real_data_tensor = torch.tensor(dataset['input_ids'][:select_size])

            # print(real_subset_output)
            real_data_tensor = torch.tensor(real_subset_output["input_ids"])

            # 检查生成的输出张量和真实数据张量的形状
            # print("Shape of generated_outputs_tensor:", generated_outputs_tensor.shape)
            print("Shape of real_data_tensor:", real_data_tensor.shape)
            
            # 使用判别器获取相似度和二分类输出
            # similarity_score, gen_logits, real_logits = discriminator_model(generated_outputs_tensor, real_data_tensor)
            similarity_score, gen_logits, real_logits = discriminator_model(generated_ids, real_data_tensor)
            
            # 计算生成器的损失
            generator_loss = train_result.metrics.get("train_loss")
            print("generator_loss: ", generator_loss)

            # 计算判别器的相似度损失
            cosine_loss = 1 - similarity_score.mean()
            print("cosine_loss: ", cosine_loss)
            
            # 计算判别器的二分类损失
            bce_loss = nn.BCELoss()
            discriminator_loss = (
                bce_loss(gen_logits.squeeze(-1), torch.zeros_like(gen_logits).squeeze(-1)) +
                bce_loss(real_logits.squeeze(-1), torch.ones_like(real_logits).squeeze(-1))
            )


            generator_losses.append(generator_loss)
            discriminator_losses.append(discriminator_loss)

            # 计算总损失
            total_loss = generator_loss + cosine_loss + discriminator_loss

            total_losses.append(total_loss)

            # 反向传播并更新模型
            trainer.optimizer.zero_grad()  # 清除生成器的梯度
            discriminator_optimizer.zero_grad()  # 清除判别器的梯度
            
            total_loss.backward()  # 计算总损失的梯度
            
            trainer.optimizer.step()  # 更新生成器
            discriminator_optimizer.step()  # 更新判别器

            # 日志记录和保存模型
            trainer.log_metrics("train", {
                "total_loss": total_loss,
                "generator_loss": generator_loss,
                "cosine_loss": cosine_loss.item(),
                "discriminator_loss": discriminator_loss.item(),
            })
            trainer.save_model()

        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            # plot_loss(training_args.output_dir, generator_losses=generator_losses, discriminator_losses=discriminator_losses, total_losses = total_losses)
            plot_loss(
                training_args.output_dir,
                generator_losses=[loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in generator_losses],
                discriminator_losses=[loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in discriminator_losses],
                total_losses=[loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in total_losses]
            )



    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset, predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)