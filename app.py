import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设置页面配置
st.set_page_config(page_title="NetGuard 网络攻击识别系统", layout="wide", initial_sidebar_state="expanded")

# 定义攻击类型映射
attack_mapping = {
    0: "正常流量",
    1: "DoS 拒绝服务攻击",
    2: "端口扫描攻击",
    3: "暴力破解攻击",
    4: "Web 应用攻击",
    5: "僵尸网络攻击",
    6: "渗透攻击",
    7: "后门攻击"
}

def main():
    # 页面标题
    st.title("🎯 NetGuard : 基于机器学习的网络攻击与后门识别系统")
    st.markdown("---")

    # 侧边栏介绍
    with st.sidebar:
        st.header("系统介绍")
        st.write("""
        这是一个针对网络流量数据的智能检测系统。
        系统支持识别 8 类流量状态，包括正常流量以及 7 种常见网络攻击。
        
        🛡️ **核心攻击识别**：
        - 后门攻击 (Backdoor)
        - DoS 攻击
        - 端口扫描
        - 暴力破解
        
        🔧 技术栈：Python, Scikit-learn, Streamlit
        """)
        st.divider()
        st.subheader("使用步骤")
        st.write("1. 准备 CSV 数据文件")
        st.write("2. 上传文件至系统")
        st.write("3. 点击「开始检测」")
        st.write("4. 查看结果与分析")

    # 主界面
    st.subheader("📁 上传流量数据")
    uploaded_file = st.file_uploader("选择 CSV 文件", type="csv")

    if uploaded_file is not None:
        # 读取数据
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head(10), use_container_width=True)

        # 加载模型 (如果存在)
        try:
            model = joblib.load("model.pkl")
            st.success("✅ 模型加载成功")
        except FileNotFoundError:
            st.warning("⚠️ 未检测到模型文件，请先运行 train.py 进行训练")
            return

        # 检测按钮
        if st.button("🚀 开始检测", type="primary"):
            with st.spinner("正在分析流量数据..."):
                # 提取特征
                X = data.drop("label", axis=1, errors="ignore")
                predictions = model.predict(X)

                # 处理结果
                result_data = data.copy()
                result_data["预测类别ID"] = predictions
                result_data["流量状态"] = [attack_mapping[pred] for pred in predictions]

                # 显示结果表格
                st.subheader("📊 检测结果详情")
                st.dataframe(result_data, use_container_width=True)

                # 统计分析
                st.subheader("📈 流量统计分析")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("总流量条数", data.shape[0])
                
                with col2:
                    attack_count = sum(1 for p in predictions if p != 0)
                    st.metric("攻击/后门流量条数", attack_count)

                # 各类别数量
                st.write("### 各类别流量分布：")
                count_series = pd.Series(predictions).value_counts().sort_index()
                for idx, count in count_series.items():
                    percentage = (count / data.shape[0]) * 100
                    st.progress(percentage/100, text=f"{attack_mapping[idx]} : {count} 条 ({percentage:.1f}%)")

                # 混淆矩阵 (如果有真实标签)
                if "label" in data.columns:
                    st.subheader("🔍 模型性能评估 - 混淆矩阵")
                    cm = confusion_matrix(data["label"], predictions)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=list(attack_mapping.values()),
                                yticklabels=list(attack_mapping.values()))
                    plt.xlabel("预测标签")
                    plt.ylabel("真实标签")
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    st.pyplot(fig)

if __name__ == "__main__":
    main()

