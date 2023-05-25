from typing import Union
from fastapi import FastAPI,  HTTPException
from dates_translator import Translator
from name_cleaner import Cleaner
import pickle
import tensorflow as tf

# LOAD ALL 


# loading tokenizers
with open('./checpoints/model_dates_checkpoint/source_tokenizer.pickle', 'rb') as handle:
    source_tokenizer = pickle.load(handle)
    
with open("./checpoints/model_dates_checkpoint/target_tokenizer.pickle", 'rb') as handle:
    target_tokenizer = pickle.load(handle)

# LOAD THE MODEL
saved_path = './checpoints/model_dates_checkpoint'
loaded = tf.saved_model.load(saved_path)
loaded_transformer = loaded.signatures['serving_default']



# INit the app
app = FastAPI()

@app.get("/api/version")
def check():
    return {"Hello": "Actual v1. Working"}


@app.get('/api/cleaner/{restaurant_name}')
async def clean_restaurant_name(restaurant_name: str):
    if len(restaurant_name) >= 256:
        raise HTTPException(
            status_code=418,
            detail="Your text is too big, please reduce it.",
        )
    try:
        cleaner = Cleaner()
        output = cleaner.clean(restaurant_name)
        return {"output":output, "error":False}
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail="some error occured",
            headers={"X-Error": str(e)},
        )


@app.get('/api/dates/translator/{text_str}')
def dates_extractor_translator(text_str:str):
    TX_SOURCE = 42
    TX_TARGET = 12
    text_str = text_str.strip()
    print(len(text_str.split()))
    if len(text_str.split()) >= 80:
        raise HTTPException(
            status_code=418,
            detail="Your text is too big, please reduce it.",
        )

    #Define the translator
    translator = Translator(
        source_tokenizer = source_tokenizer,
        target_tokenizer=target_tokenizer,
        loaded_transformer = loaded_transformer,
        TX_SOURCE=TX_SOURCE, TX_TARGET=TX_TARGET)
    
    translated, in_tokens, out_tokens, attention_weights = translator(text_str)
    return {'output':{
        "translated": translated,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "attention_weights": attention_weights[0][0].numpy().tolist()
    }, 'error':False}

