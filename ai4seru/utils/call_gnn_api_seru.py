import json
import requests

URL = "http://127.0.0.1:5001/run_deepnet"

payload = {
    "config_seru": {
        "num_of_workers": 6,
        "num_of_batches": 10,
        "max_num_of_multiple_task": 10
    },
    "problem_data": {
        "worker_to_task_dict": {
            "1": {"系数": 0.19},
            "2": {"系数": 0.23},
            "3": {"系数": 0.18},
            "4": {"系数": 0.18},
            "5": {"系数": 0.24},
            "6": {"系数": 0.24}
        },
        "worker_to_product_dict": {
            "1": {"1": 4.0, "2": 17.0, "3": 13.0, "4": 20.33333333333333, "5": 12.41689373297003},
            "2": {"1": 10.47252747252747, "2": 10.47252747252747, "3": 10.47252747252747, "4": 10.47252747252747, "5": 10.47252747252747},
            "3": {"1": 15.01335877862596, "2": 15.01335877862596, "3": 15.01335877862596, "4": 15.01335877862596, "5": 15.01335877862596},
            "4": {"1": 20.41666666666667, "2": 8.0, "3": 36.25, "4": 21.625, "5": 7.0},
            "5": {"1": 14.74922918807811, "2": 14.74922918807811, "3": 14.74922918807811, "4": 14.74922918807811, "5": 14.74922918807811},
            "6": {"1": 23.125, "2": 17.76923076923077, "3": 33.0, "4": 22.36363636363636, "5": 4.5}
        },
        "batch_to_product_dict": {
            "1": {"产品类型": "3", "批次大小": 55},
            "2": {"产品类型": "3", "批次大小": 54},
            "3": {"产品类型": "4", "批次大小": 58},
            "4": {"产品类型": "4", "批次大小": 57},
            "5": {"产品类型": "2", "批次大小": 54},
            "6": {"产品类型": "1", "批次大小": 53},
            "7": {"产品类型": "3", "批次大小": 46},
            "8": {"产品类型": "5", "批次大小": 46},
            "9": {"产品类型": "2", "批次大小": 45},
            "10": {"产品类型": "3", "批次大小": 44}
        }
    }
}

def get_gnn_result(payload):
    resp = requests.post(URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # 打印概要
    print("Keys:", list(data.keys()))
    print("reason_time:", data.get("reason_time"))

    # 只打印一小部分 edge_scores（避免刷屏）
    edge_scores = data.get("edge_scores")
    if edge_scores is not None:
        print("edge_scores shape:",
              len(edge_scores), "x", (len(edge_scores[0]) if len(edge_scores) else 0))
        print("edge_scores[0][:10]:", edge_scores[0][:10])

    # # 如需保存完整结果：
    # with open("api_result.json", "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=2)
    # print("Saved to api_result.json")
    return data

if __name__ == "__main__":
    get_gnn_result(payload)
