from transformers import pipeline

# 建立情感分析 pipeline
classifier = pipeline("sentiment-analysis")

# 測試資料
texts = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "It's okay, not great but not terrible either."
]

# 執行情感分析
results = classifier(texts)

# 輸出結果
for text, result in zip(texts, results):
    print(f"輸入：{text}")
    print(f"預測情感：{result['label']}, 信心度：{result['score']:.4f}\n")

# 中文情感分析模型
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")

# 測試資料
texts = ["這部電影真的很好看！", "這個產品太糟糕了。"]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"輸入：{text}")
    print(f"預測情感：{result['label']}, 信心度：{result['score']:.4f}\n")
