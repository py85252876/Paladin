import argparse
from dataclasses import dataclass, field
from typing import Optional, Union, Sequence, Mapping

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from peft import PeftModel

from utils import load_custom_function
from reward_funcs import reward_func


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_function_path (`str`, *required*):
            Path to the reward function file.
        reward_function_name (`str`, *required*):
            Name of the reward function in the file.
        data_files (`str`, `Sequence[str]`, or `Mapping[str, Union[str, Sequence[str]]]`, *optional*, defaults to `None`):
            Path to data files. If provided, the dataset will be loaded from the files.
    """

    reward_function_path: str = field(
        default="reward_funcs.py",
        metadata={"help": "Path to the reward function file."}
    )
    reward_function_name: str = field(
        default="reward_func",
        metadata={"help": "Name of the reward function in the file."}
    )
    data_files: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data files. If provided, the dataset will be loaded from the files."}
    )
    adapter_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trained adapter."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-trained adapter."}
    )



def main(script_args, training_args, model_args):
    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    #load the adapter from the pre-trained model
    model = PeftModel.from_pretrained(model, script_args.adapter_name, is_trainable=True)

    if script_args.checkpoint_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(
        script_args.checkpoint_dir, trust_remote_code=model_args.trust_remote_code
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
    tokenizer.pad_token = tokenizer.eos_token
    # Load reward funcs
    reward_func = load_custom_function(script_args.reward_function_path, script_args.reward_function_name)

    # Load the dataset
    if script_args.dataset_name is not None:
        dataset = load_dataset("json", data_files=script_args.dataset_name)

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=None,
        processing_class=tokenizer,
        # peft_config=get_peft_config(model_args),
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)