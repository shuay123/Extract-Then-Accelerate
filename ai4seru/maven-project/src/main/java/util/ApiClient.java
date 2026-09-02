package util;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class ApiClient {

    // 默认指向本地 Python Flask 服务地址
    private static String SERVER_URL;

    public ApiClient(String serverURL) {
        SERVER_URL = serverURL;
    }

    /**
     * 发送 HTTP POST 请求给 Python 服务
     * @return Python 返回的 JSON 字符串
     */
    public String sendRequest(String jsonPayload) {
        try {
            HttpClient client = HttpClient.newBuilder()
                    .version(HttpClient.Version.HTTP_1_1)
                    .build();

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(SERVER_URL))
                    // 算法运行时间由 config_ccea.yaml 中的 max_runtime 控制
                    // 但 Java 客户端需要一个更长的超时时间来等待响应
                    .timeout(Duration.ofMinutes(10))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                    .build();

//            System.out.println("向 Python API 发送请求...");
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            // 5. 读取响应
            if (response.statusCode() == 200) {
//                System.out.println("--- 来自 Python 结果 ---");
//                System.out.println(response.body());
                return response.body();
            } else {
                System.err.println("Python 执行失败: " + response.statusCode());
                System.err.println(response.body());
            }

        } catch (Exception e) {
            System.err.println("API 连接异常: " + e.getMessage());
            e.printStackTrace();
        }
        return null;
    }
}
