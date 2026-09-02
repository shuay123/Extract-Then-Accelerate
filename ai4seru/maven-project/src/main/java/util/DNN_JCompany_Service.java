package util;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;


import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;


public class DNN_JCompany_Service {
    //类EdgeResult，保存返回值
    public static class EdgeResult {
        public double[][] edge_scores;
        public double[][] edge_scores_batch;
        public double reason_time;
    }

    public static EdgeResult getInitialBound(SeruSamplerService.Result r, SeruSamplerService.Config cfg) {
        try {
            // 1. 准备 config_seru
            String jsonPayload = convertResultToJson_JCompany(r, cfg);

            // 3. 调用 API
            ApiClient client = new ApiClient(cfg.ServerURL);
            String json = client.sendRequest(jsonPayload);


            // 4. 解析
            if (json != null) {
                // 1. 创建 ObjectMapper
                ObjectMapper mapper = new ObjectMapper();

                // 2. 直接映射到EdgeResult 类
                EdgeResult result = mapper.readValue(json, EdgeResult.class);
                return result;
            }
        } catch (Exception e) {
            System.out.println("GNN Warning: " + e.getMessage());
        }
        EdgeResult R = null;

        return R; // 失败标记
    }

    public static String convertResultToJson_JCompany(SeruSamplerService.Result r, SeruSamplerService.Config cfg) {
        try {
            // ==========================================
            // 1. 构建 config_seru 部分
            // ==========================================
            Map<String, Object> configSeru = new LinkedHashMap<>();
            configSeru.put("num_of_workers", r.workerIds.length);
            configSeru.put("num_of_batches", r.batchIds.length);

            // Fallback values for configuration fields not exposed by Config.
            configSeru.put("max_num_of_multiple_task", 10);


            // ==========================================
            // 2. 构建 problem_data 部分
            // ==========================================
            Map<String, Object> problemData = new LinkedHashMap<>();

            // 2.1 worker_to_task_dict (多能工系数)
            // 格式: "1": { "系数": 0.1 }
            Map<String, Object> workerToTaskDict = new LinkedHashMap<>();
            for (int i = 0; i < r.workerIds.length; i++) {
                String wId = String.valueOf(i+1);
                Map<String, Double> coeffData = new HashMap<>();
                coeffData.put("系数", r.workerCoefficients[i+1]);
                workerToTaskDict.put(wId, coeffData);
            }
            problemData.put("worker_to_task_dict", workerToTaskDict);

            // 2.2 worker_to_product_dict (工人对产品类型的熟练度)
            // 格式: "1": { "1": 1.1, "2": 1.2 }
            Map<String, Map<String, Double>> workerToProductDict = new LinkedHashMap<>();

            // 初始化所有工人的 Map
            for (int wId = 0; wId < r.workerIds.length; wId++) {
                workerToProductDict.put(String.valueOf(wId+1), new LinkedHashMap<>());
            }

            // 遍历所有批次，提取产品类型，填充熟练度
            // workerProficiencies is indexed as [workerIndex][batchIndex].
            for (int j = 0; j < r.batches.size(); j++) {
                SeruSamplerService.BatchInfo batch = r.batches.get(j);
                // Convert the BatchInfo product type ID to a string.
                String productTypeId = String.valueOf(batch.productType);

                for (int i = 0; i < r.workerIds.length; i++) {
                    String wId = String.valueOf(i+1);
                    double proficiency = r.workerProficiencies[i][j];

                    // 将该[工人][产品类型]的熟练度存入
                    // 如果多个批次属于同一产品类型，后面的会覆盖前面的(理论上应相等)
                    workerToProductDict.get(wId).put(productTypeId, proficiency);
                }
            }
            problemData.put("worker_to_product_dict", workerToProductDict);

            // 2.3 batch_to_product_dict (批次属性)
            // 格式: "1": { "产品类型": "1", "批次大小": 50 }
            Map<String, Object> batchToProductDict = new LinkedHashMap<>();
            for (int j = 0; j < r.batchIds.length; j++) {
                SeruSamplerService.BatchInfo batch = r.batches.get(j);
                String bId = String.valueOf(j+1);

                Map<String, Object> batchInfo = new LinkedHashMap<>();
                batchInfo.put("产品类型", String.valueOf(batch.productType)); // JSON uses string product-type IDs.
                batchInfo.put("批次大小", r.batchSize[j+1]);

                batchToProductDict.put(bId, batchInfo);
            }
            problemData.put("batch_to_product_dict", batchToProductDict);

            // ==========================================
            // 3. 组装最终 Map 并转换为 JSON 字符串
            // ==========================================
            Map<String, Object> root = new LinkedHashMap<>();
            root.put("config_seru", configSeru);
            root.put("problem_data", problemData);

            // 创建 ObjectMapper
            ObjectMapper mapper = new ObjectMapper();
            // Pretty-print the JSON output.
            mapper.enable(SerializationFeature.INDENT_OUTPUT);

            return mapper.writeValueAsString(root);

        } catch (Exception e) {
            e.printStackTrace();
            return "{}"; // 发生错误返回空 JSON
        }
    }

}
