from pathlib import Path
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        
        # Use a shorter save directory path
        save_dir = Path(self.config.root_dir) / "samsum_ds"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save dataset to disk
        try:
            dataset_samsum_pt.save_to_disk(str(save_dir))
            logger.info(f"Dataset saved successfully at {save_dir}")
        except OSError as e:
            logger.error(f"Failed to save dataset: {e}")
            raise

# Example usage
from textSummarizer.config.configuration import ConfigurationManager

try:
    config = ConfigurationManager()
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.convert()
except Exception as e:
    raise e
