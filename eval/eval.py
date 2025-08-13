from eval import evaluate_entities,evaluate_relations
from config import config
from datasets import load_from_disk,Dataset
from model import LLM
from utils import read_from_jsonl,write_to_jsonl
from tqdm import tqdm
from format import clean_result_think


def eval_model(llm:LLM,test_path,test_label_path,flag=True):
    """

    :param llm: 模型
    :param test_path: 测试集路径
    :param test_label_path: 标签路径
    :param flag: flag==true表示不严格检查实体的类型和实体间的关系是否准确
    :return:
    """
    test_data = read_from_jsonl(test_path)
    test_labels = read_from_jsonl(test_label_path)
    precision_entity_list, recall_entity_list, f1_entity_list = [], [], []
    precision_relation_list, recall_relation_list, f1_relation_list = [], [], []
    for data, label in tqdm(zip(test_data, test_labels)):
        response = llm.generate_answer(data)
        entity_list, relation_list = clean_result_think(response)
        entity_label_list, relation_label_list = clean_result_think(label)

        if flag:
            # 非严格匹配
            entity_list = [item(0) for item in entity_list]
            entity_label_list = [item[0] for item in entity_label_list]
            relation_list = [(item[0],item[1]) for item in relation_list]
            relation_label_list = [(item[0],item[1]) for item in relation_label_list]

        p_e, r_e, f_e = evaluate_entities(entity_label_list, entity_list)
        precision_entity_list.append(p_e)
        recall_entity_list.append(r_e)
        f1_entity_list.append(f_e)

        p_r, r_r, f_r = evaluate_relations(relation_label_list, relation_list)
        precision_relation_list.append(p_r)
        recall_relation_list.append(r_r)
        f1_relation_list.append(f_r)

    precision_entity = sum(precision_entity_list) / len(precision_entity_list)
    recall_entity = sum(recall_entity_list) / len(recall_entity_list)
    f1_entity = sum(f1_entity_list) / len(f1_entity_list)

    precision_relation = sum(precision_relation_list) / len(precision_relation_list)
    recall_relation = sum(recall_relation_list) / len(recall_relation_list)
    f1_relation = sum(f1_relation_list) / len(f1_relation_list)
    print(f"Entity: precision:{precision_entity}    recall:{recall_entity}    f1:{f1_entity}")
    print(f"Relation: precision:{precision_relation}    recall:{recall_relation}    f1:{f1_relation}")
    return precision_entity,recall_entity,f1_entity,precision_relation,recall_relation,f1_relation