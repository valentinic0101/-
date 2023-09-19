# langchain
from langchain.vectorstores import Chroma
import os
# transformers
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings

embeddingMethod =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_folder='D:\\CODE_TO_RUN\\my_chatpdf\\2022_embeddings'

# 使用cpu
import torch
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda")

device=device_gpu

model_name = "THUDM/chatglm-6b"  # 预训练模型名称
local_model_path = "D:\CODE_TO_RUN\my_chatpdf\models"  # 模型的本地路径

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",cache_dir=local_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b",cache_dir=local_model_path, trust_remote_code=True).float().to(device)


# GLM回答问题
def chatmodel_glm_temperature(input_text,len_output_tokens,temperature=0.95):

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=len_output_tokens,temperature=temperature)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    response = model.process_response(response)

    return response

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float().to(device)

def chatmodel_glm_temperature(input_text,len_output_tokens,temperature=0.95,device=device_gpu):
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(**inputs, max_new_tokens=len_output_tokens,temperature=temperature)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    response = model.process_response(response)

    return response
# 获取持久存储地址
def get_persist_directory(company_name):

    company2id={'云南白药': 538, '长安汽车': 625, '长春高新': 661, '大中矿业': 1203, '宁波华翔': 2048, '紫光国微': 2049, '横店东磁': 2056, '东华软件': 2065, '雪莱特': 2076, '中材科技': 2080, '*ST新海': 2089, '*ST紫鑫': 2118, 'TCL中环': 2129, '宁波银行': 2142, '怡亚通': 2183, '全聚德': 2186, '歌尔股份': 2241, '滨江集团': 2244, '洋河股份': 2304, '杰瑞股份': 2353, '四维图新': 2405, '天虹股份': 2419, '百川股份': 2455, '中化岩土': 2542, '双星新材': 2585, '牧原股份': 2714, '网宿科技': 300017, '机器人': 300024, '红日药业': 300026, '回天新材': 300041, '星辉娱乐': 300043, '碧水源': 300070, '沃森生物': 300142, '华峰超纤': 300180, '潜能恒信': 300191, '科德教育': 300192, '舒泰神': 300204, '易华录': 300212, '上海新阳': 300236, '花园生物': 300401, '赛微电子': 300456, '普利制药': 300630, '晶瑞电材': 300655, '联合光电': 300691, '光威复材': 300699, '首都在线': 300846, '金龙鱼': 300999, '百诚医药': 301096, '雅创电子': 301099, '天宏锂电': 873152}

    company_id=company2id.get(company_name)

    str_name='db_'+str(company_id)+'_'+company_name+'_2022_hugging_embedding'

    persist_directory = os.path.join(embedding_folder, str_name)

    return persist_directory

# 找相似的12个文档
def find_similar_docs(query,company_name,persist_directory):

    print('\n ------------------------------BEGIN find_similar_docs-----------------------------\n')

    persist_directory=get_persist_directory(company_name)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddingMethod)
    docsearch=vectordb.similarity_search(query,k=12)

    docs_list=[]

    for i in range(0,len(docsearch)):

        doc=docsearch[i]
        doc=list(doc)
        docs_0=str(doc[1][1])
        print("Doc "+str(i+1)+" :\n"+docs_0+'\n\n') 

        docs_list.append(docs_0)

    print('\n ------------------------------END find_similar_docs-----------------------------\n')

    return docs_list


def answer_with_docs(query,docs_list_piece,temperature):

    print('\n ------------------------------BEGIN answer_with_docs-----------------------------\n')
    context1=' \n '.join(docs_list_piece)
    input_text1 = f"你是这家公司的发言人，需要对客户的一些问题根据公司相关信息进行解答，注意发言要以“本公司”出发。问题为“{query}”,请你根据“{context1}”回答问题，如果不能回答则收集相关信息并整理输出"
    result=chatmodel_glm_temperature(input_text1,200,temperature)

    print('\n\nAnswer:' + result + '\n')

    return result


def conclude_answers(query,result_list,temperature):

    print('\n ------------------------------BEGIN conclude_answers-----------------------------\n')
    context2=' \n '.join(result_list)
    input_text2 = f'对问题“{query}”有不同的回答“{context2}”，你需要删除重复内容，合并相似内容，汇总整理答案，每个问题只回答一次'

    final_result=chatmodel_glm_temperature(input_text2,500,temperature)


    print("总结上述结果，最终答案为： \n"+final_result)

    return final_result


