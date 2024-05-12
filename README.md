## Introduction:
My program is designed to used the past 50 years of SPY daily dataset downloaded from yahoo finance, to predict the next day's stock price. 

## Dataset handling: 
-### Input features: 
I tried to use the similar method shown in lumiwealth machine learning strategy to add RSI, MACD... indicators, together with price, pre_1_day price, pre_2.....26_day price as input features.   
-### Labels:
I used 'tomr' column as my labels/output feature. 'tomr' data is the shifted next day's 'adj_close' price data. 


## LLM: 
The llm I used is llama3 4 bit quantized model, finetuned with usloth library. 

## Current Issue:
My model, at inference stage, could not output valid content. It simply repeat the input, and output nothing. 

## Training Script
To run the training script, execute the following command:
```bash
python stock_train_llama.py
```

## Inference Script
To run the training script, execute the following command:
```bash
python stock_infer_llama.py
```
