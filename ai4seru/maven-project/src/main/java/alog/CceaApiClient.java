package alog;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class CceaApiClient {

    public static void main(String[] args) throws Exception {
        
        // 1. The Python CCEA API must be running.
        
        // 2. Load the JSON request from a file or construct it with Jackson or Gson.
        String jsonPayload = """
        {
          "config_seru": {
            "num_of_workers": 8,
            "num_of_batches": 10,
            "max_num_of_multiple_task": 10,
            "task_time": 5.0,
            "use_standard_logic": true
          },
          "problem_data": {
            "worker_to_task_dict": {
              "1": { "系数": 0.1 }, "2": { "系数": 0.2 }, "3": { "系数": 0.15 }, "4": { "系数": 0.1 },
              "5": { "系数": 0.2 }, "6": { "系数": 0.15 }, "7": { "系数": 0.1 }, "8": { "系数": 0.2 }
            },
            "worker_to_product_dict": {
              "1": { "1": 1.1, "2": 1.2 }, "2": { "1": 1.0, "2": 1.3 },
              "3": { "1": 1.2, "2": 1.1 }, "4": { "1": 1.1, "2": 1.2 },
              "5": { "1": 1.0, "2": 1.3 }, "6": { "1": 1.2, "2": 1.1 },
              "7": { "1": 1.1, "2": 1.2 }, "8": { "1": 1.0, "2": 1.3 }
            },
            "batch_to_product_dict": {
              "1": { "产品类型": "1", "批次大小": 50 }, "2": { "产品类型": "2", "批次大小": 60 },
              "3": { "产品类型": "1", "批次大小": 40 }, "4": { "产品类型": "2", "批次大小": 70 },
              "5": { "产品类型": "1", "批次大小": 55 }, "6": { "产品类型": "2", "批次大小": 65 },
              "7": { "产品类型": "1", "批次大小": 45 }, "8": { "产品类型": "2", "批次大小": 75 },
            "9": { "产品类型": "1", "批次大小": 50 }, "10": { "产品类型": "2", "批次大小": 60 }
            }
          }
        }
        """;

        HttpClient client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .build();
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://127.0.0.1:5000/run_ccea"))
                // 算法运行时间由 config_ccea.yaml 中的 max_runtime 控制
                // 但 Java 客户端需要一个更长的超时时间来等待响应
                .timeout(Duration.ofMinutes(10)) 
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                .build();

        System.out.println("向 Python API 发送请求... (这可能需要几秒到几分钟)");
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            System.out.println("--- CCEA 优化结果 (来自 Python) ---");
            System.out.println(response.body());
            // Jackson or Gson can deserialize the response into a Java DTO.
        } else {
            System.err.println("Python CCEA 执行失败: " + response.statusCode());
            System.err.println(response.body());
        }
    }
}
