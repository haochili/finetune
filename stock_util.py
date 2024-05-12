from unsloth import FastLanguageModel
from datasets import load_dataset,Dataset

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
        "unsloth/mistral-7b-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/llama-2-7b-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",
        "unsloth/gemma-7b-it-bnb-4bit",  # Instruct version of Gemma 7b
        "unsloth/gemma-2b-bnb-4bit",
        "unsloth/gemma-2b-it-bnb-4bit",  # Instruct version of Gemma 2b
        "unsloth/llama-3-8b-bnb-4bit",  # [NEW] 15 Trillion token Llama-3
    ]  # More models at https://huggingface.co/unsloth

model_name = "unsloth/llama-3-8b-bnb-4bit"
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
finetuned_model_name = 'stock_model'

# alpaca_prompt = """Make Prediction based on the Input information. The Prediction should only be one number.
#     ### Input:
#     {}

#     ### Prediction:
#     {}"""

train_prompt = """Below is an instruction that describes a stock price prediction task.\
        Input section provides features data for the stock ticker.\
        Please provide prediction on the stock price with input information.

### Instruction:
Make stock price Prediction based on the Input information of the stock. The Prediction should only be one number.

### Input:
{}

### Response:
{}"""

test_prompt = """
### Instruction:
Make stock price Prediction based on the Input information of the stock. The Prediction should only be one number.

### Input:
{}

### Response:
{}"""

class HelperUtility:
    def __init__(self,eos_token):
        self.EOS_TOKEN = eos_token

    def formatting_prompts_func_train(self,examples):
        Inputs = examples['Input']
        Outputs = examples['Output']
        texts = []
        for i in range(len(Inputs)):
            Input = Inputs[i]
            Prediction = Outputs[i]
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = train_prompt.format(Input, Prediction) + self.EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    
    def formatting_prompts_func_test(self,examples):
        Inputs = examples['Input']
        Outputs = examples['Output']
        texts = []
        for i in range(len(Inputs)):
            Input = Inputs[i]
            Prediction = Outputs[i]
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = test_prompt.format(Input, Prediction) + self.EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    def is_float(self,value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def process_input_data_csv(self,file_path, mode='train',train_pct=90):
        Inputs = []
        Outputs = []
        Labels = []
        datasets = load_dataset('csv', data_files=file_path, split={
        'train': f'train[:{train_pct}%]',
        'test': f'train[{train_pct}%:]'})
        train_dataset = datasets['train']
        test_dataset = datasets['test']
        print("Training set size:", len(train_dataset))
        print("Testing set size:", len(test_dataset))
        if mode == 'train':
            for dataset in train_dataset:
                # Create a string from the dataset, formatting floats to two decimal places, but skip 'tomr'
                Input = ', '.join(
                f'{key}: {float(value):.2f}' if isinstance(value, str) and self.is_float(value) and key != 'tomr' else
                f'{key}: {value:.2f}' if not isinstance(value, str) and key != 'tomr' else
                f'{key}: {value}'  # Handle non-numeric strings and 'tomr'
                for key, value in dataset.items() if key != 'tomr')
                Inputs.append(Input)
                Outputs.append(dataset['tomr'])
            data = {'Input': Inputs, 'Output': Outputs}
            dataset = Dataset.from_dict(data)
            #print(f"DEBUG: dataset before map:{dataset[0]}")
            dataset = dataset.map(self.formatting_prompts_func_train, batched = True,)
            #print(f"DEBUG: dataset after map:{dataset[0]}")
            return dataset
        else: 
            for dataset in test_dataset:
                Input = ', '.join(
                f'{key}: {float(value):.2f}' if isinstance(value, str) and self.is_float(value) and key != 'tomr' else
                f'{key}: {value:.2f}' if not isinstance(value, str) and key != 'tomr' else
                f'{key}: {value}'  # Handle non-numeric strings and 'tomr'
                for key, value in dataset.items() if key != 'tomr')
                Inputs.append(Input)
                Outputs.append('')
                Labels.append(dataset['tomr'])
            data = {'Input': Inputs, 'Output': Outputs}
            dataset = Dataset.from_dict(data)
            dataset = dataset.map(self.formatting_prompts_func_test, batched = True,)
            return dataset, Labels