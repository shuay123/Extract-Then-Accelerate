# python -m util.v12_1.main_api
from itertools import product as iterproduct
from util.config_loader import load_yaml_config
from model.model import UndirectedContrastiveClusteringModel
from flask import Flask, request, jsonify
import torch
import numpy as np
import time

app = Flask(__name__)

SUPPORTED_W = [15, 25]
SUPPORTED_J = [50, 100, 150, 200, 300]
TEST_J = [300]
# 模型注册表：key=(W, J)，value={'config': model, 'schedule': model, 'device': device}
MODEL_REGISTRY = {}

def reasoning(model, x, device):
    model.eval()
    start_time = time.time()
    # x = [x]
    x = torch.tensor([x], dtype=torch.float32).to(device)
    # x = x.squeeze(0)
  
 

    print(x.shape)
    edge_scores, _ = model(x)
            
            # 转换为0/1预测
    preds = (edge_scores > 0.5).float()

    edge_scores = edge_scores.squeeze(0)
    edge_scores = edge_scores.tolist()
    end_time = time.time()
    inference_time = end_time - start_time
    for i in range(len(edge_scores)):
        edge_scores[i][i] = 0
    print(f"推理时间: {inference_time:.4f} 秒")
    result = {
        'edge_scores': edge_scores,
        'reason_time': inference_time
    }

    return result


def model_init(args):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args_n = args.get('n_batch', 0) * 1
    model_config = UndirectedContrastiveClusteringModel(
        m=args.get('m', 0),
        n=args.get('n_batch', 0),
        n_nodes=args.get('n_nodes', 0),
        hidden_dim=args.get('hidden_dim', 0),
        n_gcn_layers=args.get('n_gcn_layers', 0),
        temperature=args.get('temperature', 0),
        use_contrastive=args.get('use_contrastive', False)
    ).to(device)

    model_schedule = UndirectedContrastiveClusteringModel(
        m=args.get('m', 0),
        n=args.get('n_nodes', 0),
        n_nodes=args.get('n_batch', 0),
        hidden_dim=args.get('hidden_dim', 0),
        n_gcn_layers=args.get('n_gcn_layers', 0),
        temperature=args.get('temperature', 0),
        use_contrastive=args.get('use_contrastive', False)
    ).to(device)

    model_config.load_state_dict(torch.load(args.get('best_model_path_config', 0)))
    model_schedule.load_state_dict(torch.load(args.get('best_model_path_schedule', 0)))
    # 加载最佳模型进行测试
    
    
    return model_config, model_schedule, device

def load_all_models():
    print("=" * 50)
    print("开始预加载所有 (W, J) 组合模型...")
    failed = []

    for W, J in iterproduct(SUPPORTED_W, SUPPORTED_J):
        config_name = f'JCompany_W{W}_J{J}_reasoning.yaml'
        try:
            args = load_yaml_config(config_name)
            mc, ms, dev = model_init(args)
            MODEL_REGISTRY[(W, J)] = {
                'config':   mc,
                'schedule': ms,
                'device':   dev
            }
            print(f"  ✓ W={W}, J={J} 加载成功")
        except FileNotFoundError:
            print(f"  ✗ W={W}, J={J} 跳过（配置文件不存在：{config_name}）")
            failed.append((W, J))
        except Exception as e:
            print(f"  ✗ W={W}, J={J} 加载失败：{e}")
            failed.append((W, J))

    print(f"加载完成，成功 {len(MODEL_REGISTRY)} 个，跳过 {len(failed)} 个")
    print("=" * 50)

def load_tested_models():
    print("=" * 50)
    print("开始预加载tested (W, J) 组合模型...")
    failed = []
    for W, J in iterproduct(SUPPORTED_W, TEST_J):
        config_name = f'JCompany_W{W}_J{J}_reasoning_pre.yaml'
        try:
            args = load_yaml_config(config_name)
            mc, ms, dev = model_init(args)
            MODEL_REGISTRY[(W, J, 1)] = {
                'config':   mc,
                'schedule': ms,
                'device':   dev,
                'type': 'pre'
            }
            print(f"  ✓ W={W}, J={J}, pre 加载成功")
        except FileNotFoundError:
            print(f"  ✗ W={W}, J={J}, pre 跳过（配置文件不存在：{config_name}）")
            failed.append((W, J))
        except Exception as e:
            print(f"  ✗ W={W}, J={J}, pre 加载失败：{e}")  
            failed.append((W, J))
            config_name = f'JCompany_W{W}_J{J}_reasoning_pre.yaml'
        config_name = f'JCompany_W{W}_J{J}_reasoning_f1.yaml'
        try:
            args = load_yaml_config(config_name)
            mc, ms, dev = model_init(args)
            MODEL_REGISTRY[(W, J, 2)] = {
                'config':   mc,
                'schedule': ms,
                'device':   dev,
                'type': 'f1'
            }
            print(f"  ✓ W={W}, J={J}, f1 加载成功")
        except FileNotFoundError:
            print(f"  ✗ W={W}, J={J}, f1 跳过（配置文件不存在：{config_name}）")
            failed.append((W, J))
        except Exception as e:
            print(f"  ✗ W={W}, J={J}, f1 加载失败：{e}")  
            failed.append((W, J))

    print(f"加载完成，成功 {len(MODEL_REGISTRY)} 个，跳过 {len(failed)} 个")
    print("=" * 50)

y = [[0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]]


@app.route('/run_deepnet', methods=['POST'])
def run_deepnet_endpoint():
    try:
        input_data = request.get_json(silent=True)
        if input_data is None:
            return jsonify({"error": "请求体必须是 JSON"}), 400

        config_seru_data = input_data.get("config_seru")
        problem_data = input_data.get("problem_data")
        if config_seru_data is None or problem_data is None:
            return jsonify({"error": "缺少 config_seru 或 problem_data"}), 400

        # ── 关键：从请求中读取 W 和 J ──
        W = int(config_seru_data.get("num_of_workers", 0))
        J = int(config_seru_data.get("num_of_batches", 0))
        print(f"w:{W},J:{J}")

        if (W, J, 1) not in MODEL_REGISTRY:
            available = [f"W={w},J={j}" for w, j in sorted(MODEL_REGISTRY.keys())]
            return jsonify({
                "error": f"不支持的组合 W={W}, J={J}, pre",
                "available_combinations": available
            }), 400

        # 取出对应模型
        entry = MODEL_REGISTRY[(W, J, 1)]
        mc, ms, device = entry['config'], entry['schedule'], entry['device']
        entry2 = MODEL_REGISTRY[(W, J, 2)]
        mc2, ms2, device2 = entry2['config'], entry2['schedule'], entry2['device']

        x_worker, x_batch = get_worker_to_batch2(input_data)

        result_worker1 = reasoning(mc, x_worker, device)
        result_batch1  = reasoning(ms, x_batch,  device)
        print('edge_scores_worker:',result_worker1['edge_scores'])
        # print('edge_scores_batch:',result_batch['edge_scores'])

        result_worker2 = reasoning(mc2, x_worker, device2)
        result_batch2  = reasoning(ms2, x_batch,  device2)



        return jsonify({
            'edge_scores_worker1': result_worker1['edge_scores'],
            'edge_scores_batch1':  result_batch1['edge_scores'],
            'edge_scores_worker2': result_worker2['edge_scores'],
            'edge_scores_batch2':  result_batch2['edge_scores'],
            'reason_time1': result_worker1['reason_time'] + result_batch1['reason_time'],
            'model_used1': f'W{W}_J{J}_pre',
            'model_used2': f'W{W}_J{J}_f1'   # 方便调试
        }), 200

    except Exception as e:
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
# TODO：需要改成返回x_worker & x_batch，改完了 需要测试
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
                processing_time = worker_coefficient * batch_size * product_time
            
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
    
    worker_size = len(worker_to_product)
    Max = 0
    x_worker = []
    for i in range(worker_size):
        x_i = []
        Max = max(max(worker_to_product[i]), Max)
        for j in range(len(worker_to_product[i])):
            for o in range(1):
                x_i.append(worker_to_product[i][j]/Max)
        x_worker.append(x_i)
    x_batch = [list(row) for row in zip(*x_worker)]
    return x_worker, x_batch

def get_worker_to_batch2(data):
    """
    将原始 problem_data 转换为模型所需的归一化特征矩阵 x_worker 和 x_batch
    """
    problem_data = data.get("problem_data", {})
    worker_to_task_dict = problem_data.get("worker_to_task_dict", {})
    worker_to_product_dict = problem_data.get("worker_to_product_dict", {})
    batch_to_product_dict = problem_data.get("batch_to_product_dict", {})

    # 1. 计算所有工人处理所有批次的原始时间
    # 显式排序以保证矩阵维度顺序固定 (e.g., 工人1, 2, 3... 批次1, 2, 3...)
    sorted_worker_ids = sorted(worker_to_task_dict.keys(), key=lambda x: int(x))
    sorted_batch_ids = sorted(batch_to_product_dict.keys(), key=lambda x: int(x))
    
    raw_matrix = []
    global_max = 0.0

    for w_id in sorted_worker_ids:
        worker_row = []
        worker_coefficient = worker_to_task_dict[w_id].get("系数", 1.0)
        worker_product_times = worker_to_product_dict.get(w_id, {})
        
        for b_id in sorted_batch_ids:
            batch_info = batch_to_product_dict[b_id]
            batch_product_type = str(batch_info.get("产品类型"))
            batch_size = batch_info.get("批次大小", 0)
            
            if batch_size == 0:
                processing_time = 0.0
            else:
                # 获取该工人在该产品类型上的基础工时
                product_time = worker_product_times.get(batch_product_type, 0.0)
                processing_time = batch_size * product_time
                # processing_time = worker_coefficient * batch_size * product_time
            
            worker_row.append(processing_time)
            if processing_time > global_max:
                global_max = processing_time
        
        raw_matrix.append(worker_row)

    # 2. 安全归一化处理
    # 避免除以 0
    divisor = global_max if global_max > 0 else 1.0
    
    # 构建 x_worker: [num_workers, num_batches]
    x_worker = []
    for row in raw_matrix:
        normalized_row = [val / divisor for val in row]
        x_worker.append(normalized_row)

    # 3. 构建 x_batch: [num_batches, num_workers] (即 x_worker 的转置)
    # 使用 zip(*list) 技巧进行矩阵转置
    x_batch = [list(column) for column in zip(*x_worker)]

    return x_worker, x_batch

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
    x_worker, x_batch = get_worker_to_batch(data)
    x_worker2, x_batch2 = get_worker_to_batch2(data)
    # Max = 0
    # x = []
    # for i in range(6):
    #     x_i = []
    #     Max = max(max(x1[i][:24]), Max)
    #     for j in range(len(x1[i])):
    #         for o in range(4):
    #             x_i.append(x1[i][j]/Max)
    #     x.append(x_i[:24])
    print(x_worker, x_batch)
    print(f"x_worker shape: {len(x_worker)} x {len(x_worker[0])}")
    print(f"x_batch 长度: {len(x_batch)} x {len(x_batch[0])}")


    # args = parse_args()

    y = [[0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]]
    args = load_yaml_config('JCompany_W15_J50_reasoning.yaml')
    model_config, model_schedule, device = model_init(args)

    Json_result_worker = reasoning(model_config, x_worker, y, device)
    Json_result_batch = reasoning(model_schedule, x_batch, y, device)
    Json_result = {
        'edge_scores_worker': Json_result_worker['edge_scores'],
        'edge_scores_batch': Json_result_batch['edge_scores'],
        'reason_time': Json_result_worker['reason_time'] + Json_result_batch['reason_time'],
    }
if __name__ == "__main__":
    # test()
    
    # args = load_yaml_config('JCompany_W25_J300_reasoning.yaml')
    # model_config, model_schedule, device = model_init(args)

    load_tested_models() 
    app.run(host='127.0.0.1', port=5001, debug=False)
