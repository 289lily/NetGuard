import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. 读取数据
print("正在加载数据...")
df = pd.read_csv("data.csv")
print(f"数据加载成功，共 {df.shape[0]} 条记录")

# 2. 划分特征(X)和标签(y)
X = df.drop("label", axis=1)
y = df["label"]

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 训练模型 (随机森林)
print("开始训练模型...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 模型评估
y_pred = model.predict(X_test)
print("\n===== 模型评估结果 =====")
print(f"总体准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=[
    "正常流量", "DoS攻击", "端口扫描", "暴力破解", 
    "Web攻击", "僵尸网络", "渗透攻击", "后门攻击"
]))

# 6. 保存模型
joblib.dump(model, "model.pkl")
print("\n模型训练完成并已保存为 model.pkl")
