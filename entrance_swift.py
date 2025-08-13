import json

from eval import eval_model
from config import config
from format import clean_result_think
from utils import read_from_jsonl, write_to_jsonl
from tqdm import tqdm
import torch
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForSeq2Seq, \
    TrainerCallback, TrainerState, TrainerControl
from peft import LoraConfig,get_peft_model,PeftModel

import os
from datasets import load_dataset
import swanlab
MAX_LENGTH = 2048
device = ('cuda' if torch.cuda.is_available() else 'cpu')
MAX_GENERATE_TOKENS = 2048  # 限制生成长度
MAX_INPUT_LENGTH = 2048  # 限制输入长度
def generate(tokenizer, model, message, temperature=0.7, top_p=0.9, top_k=50):
    model.eval()
    with torch.no_grad():
        # 1. 构建 prompt
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        # 2. Tokenize 输入
        inputs = tokenizer(prompt,
                           return_tensors='pt',
                           padding='longest',
                           truncation=True,
                           max_length=MAX_INPUT_LENGTH)

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 3. 生成
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=MAX_GENERATE_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05
        )

        # 4. 解码
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def eval_(test_data,test_label,tokenizer,base_model,save_path):
    # 微调前的性能计算
    entity_pred_list, relation_pred_list = [], []
    entity_gt_list, relation_gt_list = [], []
    response_list = []
    # 加载测试集数据
    for data, label in tqdm(zip(test_data, test_label),desc="eval"):
        # response = llm.generate_answer(data)
        response = generate(tokenizer, base_model, data)
        response_list.append(response)
        entity_pred_, relation_pred_ = clean_result_think(response)
        entity_gt_, relation_gt_ = clean_result_think(label)
        break
        entity_pred_list.append(entity_pred_)
        relation_pred_list.append(relation_pred_)
        entity_gt_list.append(entity_gt_)
        relation_gt_list.append(relation_gt_)
    print(f"结果写入：{save_path}")
    write_to_jsonl(response_list,save_path)
    return
    # 宽松匹配
    (precision_loose_entity, recall_loose_entity, f1_loose_entity,
     precision_loose_relation, recall_loose_relation, f1_loose_relation) = eval_model(entity_pred_list, entity_gt_list,
                                                                                      relation_pred_list,
                                                                                      relation_gt_list)
    # 严格匹配
    (precision_strict_entity, recall_strict_entity, f1_strict_entity,
     precision_strict_relation, recall_strict_relation, f1_strict_relation) = eval_model(entity_pred_list, entity_gt_list,
                                                                                      relation_pred_list,
                                                                                      relation_gt_list,False)
    return [(precision_loose_entity, recall_loose_entity, f1_loose_entity,
            precision_loose_relation, recall_loose_relation, f1_loose_relation),
            (precision_strict_entity, recall_strict_entity, f1_strict_entity,
     precision_strict_relation, recall_strict_relation, f1_strict_relation)]



class EvaluateTestCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_dataset,test_labels):
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.model = model
        self.test_labels = test_labels

    def on_evaluate(self, args, state, control, **kwargs):
        self.model.eval()

        result = eval_(self.test_dataset,self.test_labels,self.tokenizer,self.model,r'./data/result/raw_response/pre_response.jsonl')
        (precision_loose_entity, recall_loose_entity, f1_loose_entity,
         precision_loose_relation, recall_loose_relation, f1_loose_relation) = result[0]

        (precision_strict_entity, recall_strict_entity, f1_strict_entity,
         precision_strict_relation, recall_strict_relation, f1_strict_relation) = result[1]

        # 记录到 swanlab
        swanlab.log({
        "precision_loose_entity": precision_loose_entity,
        "recall_loose_entity": recall_loose_entity,
        "f1_loose_entity": f1_loose_entity,
        "precision_loose_relation":precision_loose_relation,
        "recall_loose_relation":recall_loose_relation,
        "f1_loose_relation":f1_loose_relation,

        "precision_strict_entity": precision_strict_entity,
        "recall_strict_entity":recall_strict_entity,
        "f1_strict_entity":f1_strict_entity,
        "precision_strict_relation":precision_strict_relation,
        "recall_strict_relation":recall_strict_relation,
        "f1_strict_relation":f1_strict_relation,
        })
        print(f"\nPrecision_Entity: {precision_loose_entity:.4f}, Recall_Entity: {recall_loose_entity:.4f}, F1_Entity: {f1_loose_entity:.4f}")
        print(f"\nPrecision_Entity: {precision_strict_entity:.4f}, Recall_Entity: {recall_strict_entity:.4f}, F1_Entity: {f1_strict_entity:.4f}")
        print(f"\nPrecision_Relation: {precision_loose_relation:.4f}, Recall_Relation: {recall_loose_relation:.4f}, F1_Relation: {f1_loose_relation:.4f}")
        print(f"\nPrecision_Relation: {precision_strict_relation:.4f}, Recall_Relation: {recall_strict_relation:.4f}, F1_Relation: {f1_strict_relation:.4f}")




def tokenize_train(tokenizer,example):
    messages = example["messages"]
    prompt_parts = []
    response = None
    for i, msg in tqdm(enumerate(messages),desc="tokenize"):
        role, content = msg["role"], msg["content"]
        if i == len(messages) - 1 and role == "assistant":
            response = content
        else:
            prompt_parts.append(f"<|{role}|>\n{content}")
    prompt = "\n".join(prompt_parts) + "\n<|assistant|>\n"
    model_input = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            (response or "") + tokenizer.eos_token,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
    model_input["labels"] = labels["input_ids"]
    return model_input

def tokenize_test(tokenizer,example):
    messages = example["messages"]
    prompt_parts = []

    for msg in tqdm(messages,desc='tokenize'):
        role, content = msg["role"], msg["content"]
        prompt_parts.append(f"<|{role}|>\n{content}")

    prompt = "\n".join(prompt_parts) + "\n<|assistant|>\n"

    model_input = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    model_input["prompt"] = prompt  # 可以用于后续评估可读性
    return model_input

def main():


    # 微调前测试
    model_name = config.get('llm').get('model').get('name')

    # 训练中间结果存放路径
    output_path = config.get('fine_tuning').get('output').get('path')

    train_data_path = config.get('eval').get('train').get('data_path')

    # lora adapter权重路径
    lora_adapter_path = config.get('fine_tuning').get('adapter').get('path')
    log_path = config.get('fine_tuning').get('log_path')

    test_data_path = config.get('eval').get('test').get('data_path')
    test_label_path = config.get('eval').get('test').get('label_path')

    model_name = config.get("llm").get('model').get('name')

    print("正在准备测试数据......")
    # 加载数据
    test_data = read_from_jsonl(test_data_path)
    test_label = read_from_jsonl(test_label_path)
    print("测试数据准备完成")
    print("正在加载初始模型......")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map='auto',
                                                      torch_dtype=torch.float16).eval()
    print("模型加载完成")

    print("模型微调前指标测试......")
    # 微调前
    result = eval_(test_data, test_label, tokenizer, base_model,r'./data/result/raw_response/pre_response.jsonl')
    (precision_loose_entity, recall_loose_entity, f1_loose_entity,
     precision_loose_relation, recall_loose_relation, f1_loose_relation) = result[0]

    (precision_strict_entity, recall_strict_entity, f1_strict_entity,
     precision_strict_relation, recall_strict_relation, f1_strict_relation) = result[1]

    result_list = [{
        "precision_loose_entity": precision_loose_entity,
        "recall_loose_entity": recall_loose_entity,
        "f1_loose_entity": f1_loose_entity,
        "precision_loose_relation":"precision_loose_relation",
        "recall_loose_relation":recall_loose_relation,
        "f1_loose_relation":f1_loose_relation,

        "precision_strict_entity": precision_strict_entity,
        "recall_strict_entity":recall_strict_entity,
        "f1_strict_entity":f1_strict_entity,
        "precision_strict_relation":precision_strict_relation,
        "recall_strict_relation":recall_strict_relation,
        "f1_strict_relation":f1_strict_relation
    }]
    print(f"结果:\n{result_list[0]}")
    # # 微调前的性能计算
    # entity_pred_list,relation_pred_list = [],[]
    # entity_gt_list,relation_gt_list = [],[]
    #
    # # 加载测试集数据
    # test_data = read_from_jsonl(test_data_path)
    # tst_labels = read_from_jsonl(test_label_path)
    # for data,label in tqdm(zip(test_data,tst_labels)):
    #     # response = llm.generate_answer(data)
    #     response = generate(tokenizer,base_model,data,tokenizer)
    #     entity_pred_,relation_pred_ = clean_result_think(response)
    #     entity_gt_,relation_gt_ = clean_result_think(label)
    #
    #     entity_pred_list.append(entity_pred_)
    #     relation_pred_list.append(relation_pred_)
    #     entity_gt_list.append(entity_gt_)
    #     relation_gt_list.append(relation_gt_)
    # print("宽松匹配：")
    # # 宽松匹配
    # (precision_loose_entity, recall_loose_entity, f1_loose_entity,
    #   precision_loose_relation,recall_loose_relation,f1_loose_relation) = eval_model(entity_pred_list,entity_gt_list,
    #                                                                relation_pred_list,relation_gt_list)
    # print("严格匹配：")
    # (precision_strict_entity, recall_strict_entity, f1_strict_entity,
    #  precision_strict_relation, recall_strict_relation, f1_strict_relation) = eval_model(entity_pred_list, entity_gt_list,
    #                                                                 relation_pred_list, relation_gt_list,False)

    data_files = {
        'train': train_data_path,
        'test': test_data_path
    }
    print("加载训练数据......")
    dataset = load_dataset('json',data_files=data_files)
    print("数据加载完成")
    print("训练数据转换中......")
    tokenized_train = dataset['train'].map(tokenize_train,batched=False,remove_columns=['messages'])
    print("数据转换完成")

    # tokenized_test = dataset['test'].map(tokenize_test,batched=False,remove_columns=['messages'])

    print("初始化swanlab......")

    # 登录
    api_key = os.getenv('SWANLAB_API_KEY')
    swanlab.login(api_key)
    run = swanlab.init(
        experiment_name="er_sft_llm_fine_tuning",
        description="SFT for LLM to ER"
    )
    print("初始化完成")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'],
        bias='none',
        task_type='CAUSAL_LM',
    )
    print("加载微调模型......")
    lora_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map='auto',
                                                      torch_dtype=torch.float16,
                                                      load_in_4bit=True,
                                                      quantization_config=torch.nn.Module())
    lora_model = get_peft_model(lora_model, lora_config)
    print("加载完成")

    lora_model.print_trainable_parameters()


    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_dir=log_path,
        logging_steps=10,
        save_strategy='epoch',
        save_steps=100,
        fp16=True,
        report_to=['swanlab'],
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=lora_model, padding=True)

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EvaluateTestCallback(lora_model,tokenizer,test_data,test_label)]
    )
    print("开始微调")
    trainer.train()
    print(f"lora adapter 保存在 {lora_adapter_path}")
    lora_model.save_pretrained(lora_adapter_path)

    # 加载中间层
    print("加载lora adapter......")
    fine_tuning_model = PeftModel.from_pretrained(base_model,lora_adapter_path)
    fine_tuning_model.eval()
    fine_tuning_model.to(device)

    print("在测试集上测试微调后的结果")
    result = eval_(test_data, test_label, tokenizer, fine_tuning_model,r'./data/result/raw_response/pre_response.jsonl')
    (precision_loose_entity, recall_loose_entity, f1_loose_entity,
     precision_loose_relation, recall_loose_relation, f1_loose_relation) = result[0]

    (precision_strict_entity, recall_strict_entity, f1_strict_entity,
     precision_strict_relation, recall_strict_relation, f1_strict_relation) = result[1]
    result_list.append({
        "precision_loose_entity": precision_loose_entity,
        "recall_loose_entity": recall_loose_entity,
        "f1_loose_entity": f1_loose_entity,
        "precision_loose_relation": "precision_loose_relation",
        "recall_loose_relation": recall_loose_relation,
        "f1_loose_relation": f1_loose_relation,

        "precision_strict_entity": precision_strict_entity,
        "recall_strict_entity": recall_strict_entity,
        "f1_strict_entity": f1_strict_entity,
        "precision_strict_relation": precision_strict_relation,
        "recall_strict_relation": recall_strict_relation,
        "f1_strict_relation": f1_strict_relation,
    })
    print(f"微调后的结果为：\n{result_list[1]}")
    result_path = config.get("result").get("path")
    print(f"结果保存在 {result_path} 中")
    with open(result_path,'w',encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    main()