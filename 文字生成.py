from transformers import pipeline

# 建立文字生成 pipeline
generator = pipeline("text-generation", model="gpt2")  # 小模型，下載快

# 輸入提示文字
prompt = "Once upon a time"

# 生成文字
result = generator(prompt, max_new_tokens=50)

# 輸出結果
print(result[0]['generated_text'])

from transformers import pipeline


# 建立 fill-mask pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# 輸入文字，使用 [MASK] 表示要補空的地方
text = "The capital of France is [MASK]."

# 執行補空
results = fill_mask(text)

# 輸出結果
for r in results:
    print(f"Sequence: {r['sequence']}, Score: {r['score']:.4f}, Token: {r['token_str']}")
