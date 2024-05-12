from unsloth import FastLanguageModel
from transformers import TextStreamer
from stock_util import dtype, load_in_4bit, max_seq_length, finetuned_model_name
from stock_util import HelperUtility

# Initialize the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=finetuned_model_name,  # Name of your fine-tuned model
    max_seq_length=max_seq_length,
    dtype=dtype,  # Set to None for auto-detection, or specific dtype for your GPU
    load_in_4bit=load_in_4bit,  # Reduces memory usage with quantization if True
)
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
helper_utility = HelperUtility(EOS_TOKEN)

FastLanguageModel.for_inference(model)  # Optimize model for inference

file_path = 'data/SPY_daily_data_with_indicators.csv'
datasets,labels = helper_utility.process_input_data_csv(file_path, mode='test',train_pct=90)

print('DEBUG: datasets size:',len(datasets))
print('DEBUG: labels size:',len(labels))

for formatted_prompt, label in zip(datasets,labels):
    inputs = tokenizer(formatted_prompt['text'], return_tensors="pt").to("cuda") 
    # Prepare the text streamer for generation
    text_streamer = TextStreamer(tokenizer)
    # Generate the prediction
    outcome = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1)  # Adjust max_new_tokens as needed
    print(f'outcome:{outcome}')
    # Decode the output to text
    forecasted_text = tokenizer.decode(outcome[0], skip_special_tokens=True)
    # Attempt to convert the forecasted text to a number
    print(f'forecast:{forecasted_text}')
    print('\n\n\n\n')


