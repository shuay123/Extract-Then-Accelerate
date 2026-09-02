from flask import Flask, request, jsonify 

app = Flask(__name__)

def deepnet(config_seru_data, problem_data):
    """Run the deep-learning task and return a JSON-serializable dictionary."""
    result = {
        "status": "ok",
        "some_metric": 0.95,
        "config_seru_data_echo": config_seru_data,
        "problem_data_echo": problem_data,
    }
    return result

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

        # 调用 deepnet() 
        result_dict = deepnet(config_seru_data, problem_data)

        return jsonify(result_dict), 200

    except Exception as e:
        print(f"执行 deepnet 时发生错误: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("启动 Flask 服务器，监听 http://127.0.0.1:5000")
    
    app.run(host='127.0.0.1', port=5000, debug=False)
