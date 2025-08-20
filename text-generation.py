from transformers import pipeline

# 建立文字生成 pipeline
generator = pipeline("text-generation", model="gpt2")  # 小模型，下載快

# 輸入提示文字
prompt = "Once upon a time"

# 生成文字
result = generator(prompt, max_new_tokens=50)

# 輸出結果
print(result[0]['generated_text'])


