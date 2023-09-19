# 运行需要加上 --server.fileWatcherType none
# streamlit run my_generate_stream_test.py --server.fileWatcherType none

# app.py
import streamlit as st
import json
from qa_transformer_def_for_streamlit import*
import os

import time

# PRESET
# import torch
# device_cpu = torch.device("cpu")
# device_gpu = torch.device("cuda:7")
# GPU 在函数脚本里
# END PRESET

def get_height(text):
    # 计算final_result的行数
    num_lines = len(text.split('\n'))

    # 设置留白行数，这里设置为2行
    padding_lines = 2

    # 根据行数和留白行数计算文本框的高度
    height = (num_lines + padding_lines) * 30  # 每行大约30像素高度

    return height

examples = {
    "Question 1": ("首都在线 首都在线利用自主可控的云操作系统CloudOS作为基石，建立了多个边缘算力平台，通过K3s自研算力调度系统AiDN，为实时云渲染、AI视觉、AI推理、云游戏等业务提供低时延的客户体验，以满足不同业务的需求。这方面能否具体介绍一下？", "首都在线", "尊敬的投资者：您好，公司基于k3s自研的算力调度系统AiDN, 是结合客户应用使用场景的算力调度，分3大模块“网络质量采集”、“网络流量调度”和“应用调度”可为实时云渲染、AI视觉、AI推理、云游戏等业务提供低时延的客户体验，感谢您的关注！ (来自：深交所互动易)","本公司是一家专注于提供全球云网融合服务的公司。我们致力于为游戏、音视频、电商、工具类等行业的中国企业出海客户提供计算、网络、IDC数据中心、存储和边缘计算服务。通过集成全球通讯运营商的网络资源和成熟软硬件厂商的计算、存储等设备，我们采用虚拟化、分布式计算等核心技术对 IT 实体资源进行虚拟化与产品化封装，为客户提供实时获取所需的弹性计算、存储与网络服务。我们的产品和服务包括：\n\n1. 全球云网融合服务：通过集成全球通讯运营商的网络资源和成熟软硬件厂商的计算、存储等设备，我们采用虚拟化、分布式计算等核心技术对 IT 实体资源进行虚拟化与产品化封装，为客户提供实时获取所需的弹性计算、存储与网络服务。\n\n2. 高性能GPU服务器：在产品形态上，我们提供高性能 GPU 服务器，协助客户搭建自己的渲染业务平台。\n\n3. GPU直通卡的虚拟化云主机：面向实时云渲染、AI视觉、AI推理、云游戏等业务，本公司利用自主可控的云操作系统CloudOS作为基石，建立了多个边缘算力平台，通过K3s自研算力调度系统AiDN，为实时云渲染、AI视觉、AI推理、云游戏等业务提供低时延的客户体验，以满足不同业务的需求。\n此外，本公司边缘计算在物联网中应用的领域非常广泛，特别适合具有低时延、高带宽、高可靠、海量连接、异构汇聚和本地安全隐私保护等特殊业务要求的应用场景。物联网是实现行业数字化转型的重要手段，并将催生新的产业生态和商业模式。"),
    "Question 2": ("TCL中环 新电池片项目为什么设在广州?是否存在相关利益输送?毕竟有悖常理,还请解释一下。", "TCL中环", "您好，公司项目选址整体依据具体项目建设需求、产业商业集成度及配套、支持政策等多方面因素综合研判，统筹考虑，感谢您的关注，谢谢！ (来自：深交所互动易)","新电池片项目设在广州的原因是广州拥有较为成熟和完善的电池产业链，同时广州也是中国电池产业的重要发展城市之一。此外，广州拥有优质的资源和基础设施，为新电池片项目提供了良好的发展环境和支持。\n\n关于是否存在相关利益输送，我们公司一直秉持着公正、透明、公开的原则，所有业务决策和资金运作都符合国家法律法规和公司章程的要求。我们不会存在任何利益输送的行为，也不会损害股东和投资者的利益。"),
    "Question 3": ("长安汽车 董秘您好，请问长安新能源汽车有限公司目前已经投产在售的品牌有哪些？这些品牌分别在重庆哪些工厂进行生产，我想去实地考察一下，谢谢。", "长安汽车", "您好，长安将持续坚定推进第三次创业——创新创业计划6.0版，快速落地新能源“香格里拉”计划、智能化“北斗天枢”计划，打造“新汽车 新生态”。目前，长安汽车已形成长安品牌、深蓝、阿维塔三大品牌，掌握三大新能源动力技术，四大新能源、智能化平台架构，“科技长安”形象深入人心。2023年长安汽车将陆续推出多款全新新能源产品，覆盖阿维塔、深蓝、全新OX序列、长安品牌、凯程品牌；将加快新能源产能布局，改革供应模式，全力确保每一辆车的生产，保障每一名用户的需求，在全面电动电气化转型的道路上高歌猛进，高质量奔赴新征程。感谢关注。 (来自：深交所互动易)","根据提供的信息，长安新能源汽车有限公司目前已经投产在售的品牌包括“长安 CS35”、“长安 CS75”、“长安 CS95”等。这些品牌分别在重庆不同的工厂进行生产，例如“长安 CS35”在重庆南丰工厂生产，“长安 CS75”在重庆丰和工厂生产，“长安 CS95”则在重庆长安工厂生产。此外，公司还有未投产在售的品牌，例如“长安 CS330”等。\n\n为了进行实地考察，建议前往公司位于重庆市的总部或相关工厂进行参观。公司地址为重庆市渝中区解放碑街道解放碑大道99号，电话为023-69999999，可以通过电话或官方网站了解更多相关信息。"),
    }

st.title("基于ChatGLM-6B的本地财报知识库问答系统")

st.markdown("本地化知识库检索与智能答案生成")

# Display examples as a markdown list
st.markdown("### Examples")
for index, (question, company, real_answer,generated_answer) in examples.items():
    st.markdown(f"{index}: {question}")
    st.markdown(f"- Ground Truth: `{real_answer}`")


# 创建滑动的数值选择轴
temperature = st.slider("Select the temperature", min_value=0.00, max_value=1.00, value=0.00, step=0.05)


# 创建文本输入部件
input = st.text_area("Question", height=100)
input_text=str(input)

if len(input_text.split(' '))==2:
    company_name=input.split(' ')[0]
    query=input.split(' ')[1]

    console_placeholder = st.empty()

    console_placeholder.text_area("Console", "获取本地向量库地址...", height=50)
    persist_directory = get_persist_directory(company_name)
    console_placeholder.text_area("Console", "获取成功")


    console_placeholder.text_area("Console", "检索文本中...")
    docs_list = find_similar_docs(query,company_name,persist_directory)
    console_placeholder.text_area("Console", "检索成功")

    result_list=[]
    step = 3
    for i in range(0, len(docs_list), step):
        start_idx = i
        end_idx = i + step

        docs_list_piece = '\n\n'.join(docs_list[start_idx:end_idx])
        search_output = st.text_area(f"检索段落内容【{start_idx+1}-{end_idx}】", docs_list_piece, height=400)

        answer_output = st.empty()  # 创建一个空文本区域部件
        with st.spinner(text="正在根据检索段落进行回答..."):
            result = answer_with_docs(query,docs_list[start_idx:end_idx],temperature)
            answer_output=st.text_area(f"段落【{start_idx+1}-{end_idx}】的回答: ",result,height=400)
            result_list.append(result)

    # 最终答案汇总
    final_answer_output = st.empty() 
    with st.spinner(text="正在根据4个回答进行汇总......"):
        final_result = conclude_answers(query,result_list,temperature)
        final_answer_output = st.text_area(f"根据4个回答汇总的最终回答", final_result, height=600)
