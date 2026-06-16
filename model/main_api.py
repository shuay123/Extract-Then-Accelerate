# python -m util.v12_1.main_api
from re import M
import torch
import time
import numpy as np

from util.config_loader import load_yaml_config

from model.model import UndirectedContrastiveClusteringModel

from flask import Flask, request, jsonify 
import json

app = Flask(__name__)

def reasoning(model_F, model_S, x, x_batch, device):
    """评估模型"""
    model_F.eval()
    start_time = time.time()
    # x = [x]
    x = torch.tensor([x], dtype=torch.float32).to(device)
    x_batch = torch.tensor([x_batch], dtype=torch.float32).to(device)
    # x = x.squeeze(0)
  
 

    print(x.shape)
    edge_scores, _ = model_F(x)
            
            # 转换为0/1预测
    preds = (edge_scores > 0.5).float()

    edge_scores = edge_scores.squeeze(0)
    edge_scores = edge_scores.tolist()

    edge_scores_batch, _ = model_S(x_batch)
            
            # 转换为0/1预测
    preds = (edge_scores_batch > 0.5).float()

    edge_scores_batch = edge_scores_batch.squeeze(0)
    edge_scores_batch = edge_scores_batch.tolist()


    end_time = time.time()
    inference_time = end_time - start_time
    for i in range(len(edge_scores)):
        edge_scores[i][i] = 0
    print(f"推理时间: {inference_time:.4f} 秒")
    result = {
        'edge_scores': edge_scores,
        'edge_scores_batch': edge_scores_batch,
        'reason_time':inference_time
    }
    print(result)

    return result


def model_init(args):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args_n = args.get('n_batch', 0) * 4
    model_F = UndirectedContrastiveClusteringModel(
        m=args.get('m', 0),
        n=args_n,
        n_nodes=args.get('n_nodes', 0),
        hidden_dim=args.get('hidden_dim', 0),
        n_gcn_layers=args.get('n_gcn_layers', 0),
        temperature=args.get('temperature', 0),
        use_contrastive=args.get('use_contrastive', False)
    ).to(device)

    model_F.load_state_dict(torch.load(args.get('best_model_path', 0)))
    args_n = args.get('n_nodes', 0) * 4
    model_S = UndirectedContrastiveClusteringModel(
        m=args.get('m', 0),
        n=args_n,
        n_nodes=args.get('n_batch', 0),
        hidden_dim=args.get('hidden_dim', 0),
        n_gcn_layers=args.get('n_gcn_layers', 0),
        temperature=args.get('temperature', 0),
        use_contrastive=args.get('use_contrastive', False)
    ).to(device)

    model_S.load_state_dict(torch.load(args.get('best_model_path_schedule', 0)))
    
    # 加载最佳模型进行测试
    
    
    return model_F, model_S, device
y = [[0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]]


@app.route('/run_deepnet', methods=['POST'])
def run_deepnet_endpoint():
    print("收到 /run_deepnet 请求...")
    try:
        input_data = request.get_json(silent=True)
        if input_data is None:
            return jsonify({"error": "请求体必须是 JSON"}), 400

        config_seru_data = input_data.get("config_seru")
        problem_data = input_data.get("problem_data")
        if config_seru_data is None or problem_data is None:
            # 用 is None 避免空 dict 也被当成 False
            return jsonify({"error": "缺少 config_seru 或 problem_data"}), 400

        x, x_batch= get_worker_to_batch(input_data)
        print(x)
        # 调用 deepnet() 
        print("\n开始推理...")
        Json_result = reasoning(model_F, model_S, x, x_batch, device)
        

        return jsonify(Json_result), 200

    except Exception as e:
        print(f"执行 deepnet 时发生错误: {e}")
        return jsonify({"error": str(e)}), 500
def get_data(input_data):
    config_seru_data = input_data.get("config_seru")
    problem_data = input_data.get("problem_data")
    if config_seru_data is None or problem_data is None:
        # 用 is None 避免空 dict 也被当成 False
        return jsonify({"error": "缺少 config_seru 或 problem_data"}), 400
    
    # 3. 预检入参一致性
    print("正在初始化 CCEA 实例...")
    try:
        num_batches = int(config_seru_data.get("num_of_batches"))
    except Exception:
        return jsonify({"error": "config_seru.num_of_batches 必须为整数"}), 400

    batch_dict = problem_data.get("batch_to_product_dict", {})
    # 统一将可转换的键转为整数进行范围校验
    key_set = set()
    for k in batch_dict.keys():
        try:
            key_set.add(int(k))
        except Exception:
            # 非数字键忽略参与范围判断，但会在加载阶段抛出更详细错误
            pass
    expected = set(range(1, num_batches + 1))
    missing_ids = sorted(list(expected - key_set))
    if missing_ids:
        return jsonify({
            "error": "batch_to_product_dict 未完整覆盖所需批次ID",
            "num_of_batches": num_batches,
            "missing_batch_ids": missing_ids[:50]
        }), 400

    return config_seru_data, problem_data
def get_worker_to_batch(data):
    worker_to_task_dict = data["problem_data"]["worker_to_task_dict"]
    worker_to_product_dict = data["problem_data"]["worker_to_product_dict"]
    batch_to_product_dict = data["problem_data"]["batch_to_product_dict"]

    processing_times = {}

    for worker_id in worker_to_task_dict:
        processing_times[worker_id] = {}
        worker_coefficient = worker_to_task_dict[worker_id]["系数"]
        worker_product_times = worker_to_product_dict[worker_id]
        
        for batch_id in batch_to_product_dict:
            batch_product_type = batch_to_product_dict[batch_id]["产品类型"]
            batch_size = batch_to_product_dict[batch_id]["批次大小"]
            
            if batch_size == 0:
                processing_time = 0
            else:
                product_time = worker_product_times[batch_product_type]
                processing_time = batch_size * product_time
            
            processing_times[worker_id][batch_id] = processing_time

    # 保存结果
    worker_to_product = []
    print("每个工人处理每个批次的时间：")
    for worker_id, batch_times in processing_times.items():
        print(f"工人 {worker_id}:")
        worker_i_to_product= []
        for batch_id, time in batch_times.items():
            print(f"  批次 {batch_id}: {time:.2f}")
            worker_i_to_product.append(time)
        worker_to_product.append(worker_i_to_product)
    # worker_to_product =  worker_to_product.to_numpy(dtype=float)
    worker_size = len(worker_to_product)
    Max = 0
    for i in range(worker_size):
        Max = max(max(worker_to_product[i]), Max)
    worker_to_product=np.array(worker_to_product)/Max
    # x1 = []
    # for i in range(worker_size):
    #     x_i = []
    #     for j in range(len(worker_to_product[i])):
    #         for o in range(4):
    #             x_i.append(worker_to_product[i][j])
    #     x1.append(x_i)
    x = []
    for w_i in worker_to_product:
        w4 = np.repeat(w_i,4)
        x.append(w4)
    batch_to_worker = worker_to_product.transpose()
    x_batches = []
    for B_i in batch_to_worker:
        b4 = np.repeat(B_i,4)
        x_batches.append(b4)
    print("X:",x)
    print("X_batches:",x_batches)
    return x, x_batches


def test():
    data = {
  "config_seru" : {
    "num_of_workers" : 6,
    "num_of_batches" : 10,
    "max_num_of_multiple_task" : 10
  },
  "problem_data" : {
    "worker_to_task_dict" : {
      "1" : {
        "系数" : 0.19
      },
      "2" : {
        "系数" : 0.23
      },
      "3" : {
        "系数" : 0.18
      },
      "4" : {
        "系数" : 0.18
      },
      "5" : {
        "系数" : 0.24
      },
      "6" : {
        "系数" : 0.24
      }
    },
    "worker_to_product_dict" : {
      "1" : {
        "3" : 13.0,
        "4" : 20.33333333333333,
        "2" : 17.0,
        "1" : 4.0,
        "5" : 12.41689373297003
      },
      "2" : {
        "3" : 10.47252747252747,
        "4" : 10.47252747252747,
        "2" : 10.47252747252747,
        "1" : 10.47252747252747,
        "5" : 10.47252747252747
      },
      "3" : {
        "3" : 15.01335877862596,
        "4" : 15.01335877862596,
        "2" : 15.01335877862596,
        "1" : 15.01335877862596,
        "5" : 15.01335877862596
      },
      "4" : {
        "3" : 36.25,
        "4" : 21.625,
        "2" : 8.0,
        "1" : 20.41666666666667,
        "5" : 7.0
      },
      "5" : {
        "3" : 14.74922918807811,
        "4" : 14.74922918807811,
        "2" : 14.74922918807811,
        "1" : 14.74922918807811,
        "5" : 14.74922918807811
      },
      "6" : {
        "3" : 33.0,
        "4" : 22.36363636363636,
        "2" : 17.76923076923077,
        "1" : 23.125,
        "5" : 4.5
      }
    },
    "batch_to_product_dict" : {
      "1" : {
        "产品类型" : "3",
        "批次大小" : 55
      },
      "2" : {
        "产品类型" : "3",
        "批次大小" : 54
      },
      "3" : {
        "产品类型" : "4",
        "批次大小" : 58
      },
      "4" : {
        "产品类型" : "4",
        "批次大小" : 57
      },
      "5" : {
        "产品类型" : "2",
        "批次大小" : 54
      },
      "6" : {
        "产品类型" : "1",
        "批次大小" : 53
      },
      "7" : {
        "产品类型" : "3",
        "批次大小" : 46
      },
      "8" : {
        "产品类型" : "5",
        "批次大小" : 46
      },
      "9" : {
        "产品类型" : "2",
        "批次大小" : 45
      },
      "10" : {
        "产品类型" : "3",
        "批次大小" : 44
      }
    }
  }
}
    x, x_batch = get_worker_to_batch(data)
    # Max = 0
    # x = []
    # for i in range(6):
    #     x_i = []
    #     Max = max(max(x1[i][:24]), Max)
    #     for j in range(len(x1[i])):
    #         for o in range(4):
    #             x_i.append(x1[i][j]/Max)
    #     x.append(x_i[:24])
    print(x)


    # args = parse_args()

    # y = [[0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 1, 0],
    #     [0, 1, 0, 1, 1, 0],
    #     [0, 1, 1, 0, 1, 0],
    #     [0, 1, 1, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0]]
    args = load_yaml_config('JCompany_W6_J10_reasoning.yaml')
    model_F, model_S, device = model_init(args)

    Json_result = reasoning(model_F, model_S, x, x_batch, device)
    print(Json_result)
if __name__ == "__main__":
    # test()
    
    args = load_yaml_config('JCompany_W7_J12_reasoning.yaml')
    model_F, model_S, device = model_init(args)


    app.run(host='127.0.0.1', port=5001, debug=False)
