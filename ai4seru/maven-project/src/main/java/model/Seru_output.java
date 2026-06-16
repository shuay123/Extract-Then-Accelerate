package model;

import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import util.ExcelMultiSheetWriter;
import util.SeruSamplerService;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;


public class Seru_output {
    // ====== 打印解决方案 ======
    public static void printSolution(Seru.Result r) {
        System.out.println("\n==============================");
        System.out.println("全局Cmax = " + String.format("%.2f", r.Cmax));
        System.out.println("对应赛汝构造: " + r.config);
        System.out.println("批次 -> 赛汝 分配：");
        for (int m = 1; m <= r.M; m++) {
            System.out.printf("  Batch %d -> Seru %d%n", m, r.batchToSeru[m]);
        }

        // 按赛汝汇总
        Map<Integer, List<Integer>> seruBatches = new LinkedHashMap<>();
        for (int j = 1; j <= r.J; j++) seruBatches.put(j, new ArrayList<>());
        for (int m = 1; m <= r.M; m++) seruBatches.get(r.batchToSeru[m]).add(m);
        System.out.println("按赛汝汇总：");
        for (int j = 1; j <= r.J; j++) {
            System.out.println("  Seru " + j + ": " + seruBatches.get(j));
        }
    }

    // ====== 打印最优解 ======
    public static void printBestSolution(Seru.Result best) {
        if (best != null) {
            System.out.println("\n==============================");
            System.out.println("全局最优 Cmax = " + String.format("%.2f", best.Cmax));
            System.out.println("对应赛汝构造: " + best.config);
            printSolution(best);
        } else {
            System.out.println("未得到可行解。");
        }
    }

    // ====== 保存seru解 ======
    //outpath = "dateset/2020_config/"
    public static void ExportToExcel_SeruResult(String outpath ,int problemIndex, Seru.Result res) {

        Map<String, ExcelMultiSheetWriter.DataTable> Sheets = new HashMap<>();

        List<String> problemHeaders = Arrays.asList(
                "J", "W", "index"
        );
        List<List<Object>> problemRows = new ArrayList<>();
        List<Object> row1 = new ArrayList<>();
        row1.add(res.problem.M);
        row1.add(res.problem.W);
        row1.add(res.problem.index);
        problemRows.add(row1);

        Sheets.put("problem", ExcelMultiSheetWriter.DataTable.of(
                problemHeaders,
                problemRows
        ));

        List<String> SeruHeaders = Arrays.asList(
                "idx", "Cmax", "value"
        );
        List<List<Object>> SeruRows = new ArrayList<>();
        List<Object> row2 = new ArrayList<>();
        row2.add(res.idx);
        row2.add(res.Cmax);
        row2.add(res.value);
        SeruRows.add(row2);

        Sheets.put("result_value", ExcelMultiSheetWriter.DataTable.of(SeruHeaders, SeruRows));

        List<String> SeruConfigHeaders = Arrays.asList(
                "SeruID"
        );

        List<List<Object>> SeruConfigRows = new ArrayList<>();
        for (int i = 0; i <res.config.serus.size(); i++){

            List<Object> row = new ArrayList<>();
            row.add(res.config.serus.get(i));
            SeruConfigRows.add(row);
        }
        Sheets.put("SeruConfig", ExcelMultiSheetWriter.DataTable.of(SeruConfigHeaders, SeruConfigRows));

//        List<String> SeruBatchHeaders = Arrays.asList(
//                "SeruBatch", "BatchID"
//        );
//        List<List<Object>> SeruBatchRows = new ArrayList<>();
//        for (int i = 0; i <res.batchToSeru.length; i++){
//            List<Object> row = new ArrayList<>();
//            row.add(res.batchToSeru[i]);
//            row.add(i);
//            SeruBatchRows.add(row);
//        }
//        Sheets.put("SeruBatch", ExcelMultiSheetWriter.DataTable.of(SeruBatchHeaders, SeruBatchRows));
        double[][][] worker_T_norm = Get_T_Norm(res.problem);
        double[][] worker_basetime_norm = Get_BaseTime_Norm(res.problem);
        Seru.SeruProblem Seru_p = res.problem;
        List<String> WorkerFeatureHeaders = Arrays.asList(
                "WorkerID", "Features"
        );
        List<List<Object>> WorkerFeatureRows = new ArrayList<>();
        for(int w = 1; w <= Seru_p.W; w++){
            List<Object> row = new ArrayList<>();
            row.add(w);
            for (int m = 1; m <= Seru_p.M; m++){
                for (int o = 1; o <= Seru_p.O; o++){
                    row.add(worker_T_norm[w][m][o]);
                }
            }
            for (int m = 1; m <= Seru_p.M; m++){
                row.add(worker_basetime_norm[w][m]);
            }
            WorkerFeatureRows.add(row);
        }
        Sheets.put("workerFeature", ExcelMultiSheetWriter.DataTable.of(WorkerFeatureHeaders, WorkerFeatureRows));

        try {
            ExcelMultiSheetWriter.writeXlsx(Sheets, outpath+ problemIndex+"_"+res.idx +".xlsx");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // ====== 保存seru解 ======
    //outpath = "dateset/2020_config/"
    public static void ExportToExcel_SeruResult_JCompany(String outpath ,int problemIndex, Seru.Result[] results) {


        Seru.Result res = results[0];
        Map<String, ExcelMultiSheetWriter.DataTable> Sheets = new HashMap<>();

        List<String> problemHeaders = Arrays.asList(
                "index","idx","J", "W", "solving_time"
        );
        List<List<Object>> problemRows = new ArrayList<>();
        List<Object> row1 = new ArrayList<>();
        row1.add(res.problem.index);
        row1.add(res.idx);
        row1.add(res.problem.M);
        row1.add(res.problem.W);
        row1.add(res.problem.solving_time);

        problemRows.add(row1);

        Sheets.put("problem", ExcelMultiSheetWriter.DataTable.of(
                problemHeaders,
                problemRows
        ));

        List<String> SeruHeaders = Arrays.asList(
                "idx", "SeruConfig","value", "Cmax","batchtoSeru","1","2","3","4","5","6","7","8","9","10"
        );
        List<List<Object>> SeruRows = new ArrayList<>();

        for (int i = 0; i < res.problem.ResultofConfigs.length; i++){
            List<Object> row2 = new ArrayList<>();
            row2.add(res.problem.ResultofConfigs[i][0]);
            row2.add(res.problem.ResultofConfigs[i][3].toString());
            row2.add(res.problem.ResultofConfigs[i][1]);
            row2.add(res.problem.ResultofConfigs[i][2]);
            for(int j = 0;j < res.problem.M+1; j++) row2.add(results[i].batchToSeru[j]);
            SeruRows.add(row2);
        }

        Sheets.put("result_value", ExcelMultiSheetWriter.DataTable.of(SeruHeaders, SeruRows));

        List<String> ValuesHeaders = Arrays.asList(
                "values"
        );

        List<List<Object>> valuesRows = new ArrayList<>();
        for (int j = 0; j <res.problem.value.length; j++){
            List<Object> row = new ArrayList<>();
            for(int i = 0; i < res.problem.value.length; i++){
                row.add(res.problem.value[j][i]);
            }
            valuesRows.add(row);
        }
        Sheets.put("Values", ExcelMultiSheetWriter.DataTable.of(ValuesHeaders, valuesRows));

        List<String> LabelsHeaders = Arrays.asList(
                "Labels"
        );

        List<List<Object>> labelsRows = new ArrayList<>();
        for (int j = 0; j <res.problem.label.length; j++){
            List<Object> row = new ArrayList<>();
            for(int i = 0; i < res.problem.label.length; i++){
                row.add(res.problem.label[j][i]);
            }
            labelsRows.add(row);
        }
        Sheets.put("Labels", ExcelMultiSheetWriter.DataTable.of(LabelsHeaders, labelsRows));

        List<String> WokersFHeaders = Arrays.asList(
                "WokersFeature"
        );
        double[][] WR = new double[res.problem.W][res.problem.M];
        double max = 0;
        for (int j = 1; j <res.problem.baseTime_mi[0].length; j++){
            for(int i = 1; i < res.problem.baseTime_mi.length; i++){
                WR[j-1][i-1]=res.problem.baseTime_mi[i][j]*res.problem.batchSize[i];
                max = Math.max(max,WR[j-1][i-1]);
            }
        }
        for (int j = 1; j <res.problem.baseTime_mi[0].length; j++){
            for(int i = 1; i < res.problem.baseTime_mi.length; i++){
                WR[j-1][i-1]=WR[j-1][i-1]/max;
            }
        }
        List<List<Object>> WokersFRows = new ArrayList<>();
        for (int j = 1; j <res.problem.baseTime_mi[0].length; j++){
            List<Object> row = new ArrayList<>();
            for(int i = 1; i < res.problem.baseTime_mi.length; i++){
                row.add(WR[j-1][i-1]);
            }
            WokersFRows.add(row);
        }
        Sheets.put("WokersFeature", ExcelMultiSheetWriter.DataTable.of(WokersFHeaders, WokersFRows));


//        List<String> SeruBatchHeaders = Arrays.asList(
//                "SeruBatch", "BatchID"
//        );
//        List<List<Object>> SeruBatchRows = new ArrayList<>();
//        for (int i = 0; i <res.batchToSeru.length; i++){
//            List<Object> row = new ArrayList<>();
//            row.add(res.batchToSeru[i]);
//            row.add(i);
//            SeruBatchRows.add(row);
//        }
//        Sheets.put("SeruBatch", ExcelMultiSheetWriter.DataTable.of(SeruBatchHeaders, SeruBatchRows));
        double[][][] worker_T_norm = Get_T_Norm(res.problem);
        double[][] worker_basetime_norm = Get_BaseTime_Norm(res.problem);
        Seru.SeruProblem Seru_p = res.problem;
        List<String> WorkerFeatureHeaders = Arrays.asList(
                "WorkerID", "Features"
        );
        List<List<Object>> WorkerFeatureRows = new ArrayList<>();
        for(int w = 1; w <= Seru_p.W; w++){
            List<Object> row = new ArrayList<>();
            row.add(Seru_p.workerIds[w-1]);
            for (int m = 1; m <= Seru_p.M; m++){
                for (int o = 1; o <= Seru_p.O; o++){
                    row.add(worker_T_norm[w][m][o]);
                }
            }
            for (int m = 1; m <= Seru_p.M; m++){
                row.add(worker_basetime_norm[w][m]);
            }
            WorkerFeatureRows.add(row);
        }
        Sheets.put("workerFeatureOriginal", ExcelMultiSheetWriter.DataTable.of(WorkerFeatureHeaders, WorkerFeatureRows));

        List<String> batchInfoHeaders = Arrays.asList(
                "BatchID", "Type","size"
        );
        List<List<Object>> batchInfoRows = new ArrayList<>();
        for (int i = 0; i <Seru_p.batchInfo.length; i++){
            List<Object> row = new ArrayList<>();
            row.add(Seru_p.batchInfo[i][0]);
            row.add(Seru_p.batchInfo[i][1]);
            row.add(Seru_p.batchInfo[i][2]);
            batchInfoRows.add(row);
        }
        Sheets.put("batchInfo", ExcelMultiSheetWriter.DataTable.of(batchInfoHeaders, batchInfoRows));

        try {
            ExcelMultiSheetWriter.writeXlsx(Sheets, outpath+ problemIndex+".xlsx");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static void ExportToExcel_SeruResult_JCompany_Inpaper(String outpath , int problemIndex, Seru.Result[] results, SeruSamplerService.Result r) {


        Seru.Result res = results[0];
        Map<String, ExcelMultiSheetWriter.DataTable> Sheets = new HashMap<>();

        List<String> problemHeaders = Arrays.asList(
                "index","J", "W", "solving_time(ms)","Cmax"
        );
        List<List<Object>> problemRows = new ArrayList<>();
        List<Object> row1 = new ArrayList<>();
        row1.add(res.problem.index);
        row1.add(res.problem.M);
        row1.add(res.problem.W);
        row1.add(res.problem.solving_time);
        row1.add(res.Cmax);

        problemRows.add(row1);

        Sheets.put("problem", ExcelMultiSheetWriter.DataTable.of(
                problemHeaders,
                problemRows
        ));

        List<String> SeruHeaders = Arrays.asList(
                "idx", "SeruConfig", "Cmax","batchtoSeru","1","2","3","4","5","6","7","8","9","10","11","12"
        );
        List<List<Object>> SeruRows = new ArrayList<>();
        List<Object> row2 = new ArrayList<>();
        row2.add(res.idx);
        row2.add(res.config.toString());
        row2.add(res.Cmax);
        for(int j = 0;j < res.problem.M+1; j++) row2.add(res.batchToSeru[j]);
        SeruRows.add(row2);

        Sheets.put("result_value", ExcelMultiSheetWriter.DataTable.of(SeruHeaders, SeruRows));

        Seru.SeruProblem Seru_p = res.problem;


        List<String> batchInfoHeaders = Arrays.asList(
                "BatchID", "Type","size"
        );
        List<List<Object>> batchInfoRows = new ArrayList<>();
        for (int i = 0; i <Seru_p.batchInfo.length; i++){
            List<Object> row = new ArrayList<>();
            row.add(Seru_p.batchInfo[i][0]);
            row.add(Seru_p.batchInfo[i][1]);
            row.add(Seru_p.batchInfo[i][2]);
            batchInfoRows.add(row);
        }
        Sheets.put("batchInfo", ExcelMultiSheetWriter.DataTable.of(batchInfoHeaders, batchInfoRows));

        List<String> workerInfoHeaders = Arrays.asList(
                "workerID\\Type", "1","2","3","4","5","Coefficients"
        );
        List<List<Object>> workerInfoRows = new ArrayList<>();
        for (int i = 0; i <r.workerIds.length; i++){
            List<Object> row = new ArrayList<>();
            row.add(r.workerIds[i]);
            for (int j = 0; j<5; j++) row.add(r.workerProficienciesToType[i][j]);
            row.add(r.workerCoefficients[i+1]);
            workerInfoRows.add(row);
        }
        Sheets.put("workerInfo", ExcelMultiSheetWriter.DataTable.of(workerInfoHeaders, workerInfoRows));

        try {
            ExcelMultiSheetWriter.writeXlsx(Sheets, outpath+ problemIndex+".xlsx");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static void ExportToExcel_SeruResult_JCompany_Inpaper_con(String outpath , Object[][] resultsAll) {

        Map<String, ExcelMultiSheetWriter.DataTable> Sheets = new HashMap<>();

        List<String> resultHeaders = Arrays.asList(
                "InstanceID","flag_e","MIP","","","GNN","","","","","","","GNN2","","","","","","",""
        );
        List<List<Object>> resultRows = new ArrayList<>();
        List<Object> row = new ArrayList<>();
        // NSF:seru的formation数量，NCP：conflictPairsNum
        String[] title2={"","","Cmax", "NSF","SolvingTime","Cmax", "NSF","SolvingTime","SolvingTimeMIP","SolvingTimeGNN","gap","speedup","Cmax", "NSF","SolvingTime","SolvingTimeMIP","SolvingTimeGNN","gap","speedup","NCP"};
        Collections.addAll(row, title2);
        resultRows.add(row);
        for (int i = 0; i <resultsAll.length-2; i++){
            List<Object> row1 = new ArrayList<>();
            for(int j = 0; j <resultsAll[0].length;j++) row1.add(resultsAll[i][j]);

            resultRows.add(row1);
        }
        Sheets.put("result", ExcelMultiSheetWriter.DataTable.of(resultHeaders, resultRows));

        List<String> AllHeaders = Arrays.asList();

        List<List<Object>> AllRows = new ArrayList<>();
        for (int i = 2; i > 0; i--) {
            List<Object> row1 = new ArrayList<>();
            row1.add(resultsAll[resultsAll.length-i][0]);
            AllRows.add(row1);
        }
        Sheets.put("All", ExcelMultiSheetWriter.DataTable.of(AllHeaders, AllRows));

        try {
            ExcelMultiSheetWriter.writeXlsx(Sheets, outpath+ "result.xlsx");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static void ExportToExcel_SeruResult_JCompany_Inpaper_con_alpha(String outpath , Object[][] resultsAll) {

        Map<String, ExcelMultiSheetWriter.DataTable> Sheets = new HashMap<>();

        List<String> resultHeaders = Arrays.asList(
                "Alpha","flag_e","MIP_Cmax","MIP_NSF","MIP_SolvingTime",
                "GNN_Cmax","gap","GNN_NSF","gap","GNN_SolvingTime","gap","SolvingTimeMIP","SolvingTimeGNN","speedup"
        );
        List<List<Object>> resultRows = new ArrayList<>();
        List<Object> row = new ArrayList<>();
        // NSF:seru的formation数量，NCP：conflictPairsNum
        for (Object[] objects : resultsAll) {
            List<Object> row1 = new ArrayList<>(Arrays.asList(objects).subList(0, resultsAll[0].length));

            resultRows.add(row1);
        }
        Sheets.put("result", ExcelMultiSheetWriter.DataTable.of(resultHeaders, resultRows));

        try {
            ExcelMultiSheetWriter.writeXlsx(Sheets, outpath+ "resultAlpha.xlsx");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    // 辅助：写一整行字符串表头
    private static void writeRow(org.apache.poi.ss.usermodel.Sheet sheet, int rowIndex, String[] values) {
        org.apache.poi.ss.usermodel.Row row = sheet.createRow(rowIndex);
        for (int i = 0; i < values.length; i++) {
            row.createCell(i).setCellValue(values[i]);
        }
    }

    // 辅助：根据类型设置单元格值
    private static void setCellValue(org.apache.poi.ss.usermodel.Cell cell, Object value) {
        if (value == null) {
            cell.setCellValue("");
        } else if (value instanceof Number) {
            cell.setCellValue(((Number) value).doubleValue());
        } else if (value instanceof Boolean) {
            cell.setCellValue((Boolean) value);
        } else {
            cell.setCellValue(value.toString());
        }
    }


    private static void WorkerTOworkerExportToExcel(String name,
                                                    Seru.Result res,
                                                    Map<String,
                                                            ExcelMultiSheetWriter.DataTable> Sheets,
                                                    int[] workerIDTOworkerID) {


        Map<String, ExcelMultiSheetWriter.DataTable> problemSheets = new HashMap<>();

        List<String> problemHeaders = Arrays.asList(
                "J", "W", "index"
        );
        List<List<Object>> problemRows = new ArrayList<>();
        List<Object> row1 = new ArrayList<>();
        row1.add(res.problem.M);
        row1.add(res.problem.W);
        row1.add(res.problem.index);
        problemRows.add(row1);

        problemSheets.put("problem", ExcelMultiSheetWriter.DataTable.of(
                problemHeaders,
                problemRows
        ));

        List<String> SeruHeaders = Arrays.asList(
                "idx", "Cmax", "value"
        );
        List<List<Object>> SeruRows = new ArrayList<>();
        List<Object> row2 = new ArrayList<>();
        row2.add(res.idx);
        row2.add(res.Cmax);
        row2.add(res.value);
        SeruRows.add(row2);

        Sheets.put("result_value", ExcelMultiSheetWriter.DataTable.of(SeruHeaders, SeruRows));

        List<String> SeruConfigHeaders = Arrays.asList(
                "SeruID"
        );

        List<List<Object>> SeruConfigRows = new ArrayList<>();
        for (int i = 0; i <res.config.serus.size(); i++){

            List<Object> row = new ArrayList<>();
            row.add(res.config.serus.get(i));
            SeruConfigRows.add(row);
        }
        Sheets.put("SeruConfig", ExcelMultiSheetWriter.DataTable.of(SeruConfigHeaders, SeruConfigRows));

        List<String> SeruBatchHeaders = Arrays.asList(
                "BatchID","SeruBatch"
        );
        List<List<Object>> SeruBatchRows = new ArrayList<>();
        for (int i = 0; i <res.batchToSeru.length; i++){
            List<Object> row = new ArrayList<>();
            row.add(i);
            row.add(res.batchToSeru[i]);
            SeruBatchRows.add(row);
        }
        Sheets.put("SeruBatch", ExcelMultiSheetWriter.DataTable.of(SeruBatchHeaders, SeruBatchRows));

        List<String> SeruWorkerHeaders = Arrays.asList(
                "WorkerID","ToWorkerID","label"
        );

        List<List<Object>> SeruWorkerRows = new ArrayList<>();
        List<Object> row = new ArrayList<>();
        row.add(workerIDTOworkerID[0]);
        row.add(workerIDTOworkerID[1]);
        row.add(workerIDTOworkerID[2]);
        SeruWorkerRows.add(row);

        Sheets.put("SeruWorkerTOworker", ExcelMultiSheetWriter.DataTable.of(SeruWorkerHeaders, SeruWorkerRows));

        try {
            ExcelMultiSheetWriter.writeXlsx(Sheets, "dateset/2020_worker/"+ name +".xlsx");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static double[][][] Get_T_Norm(Seru.SeruProblem Seru_p){
        double[][][] T_norm = new double[Seru_p.O + 1][Seru_p.M + 1][Seru_p.W + 1];
        // 填充 T 矩阵
        double max = 0;
        for (int o = 1; o <= Seru_p.O; o++) {
            for (int m = 1; m <= Seru_p.M; m++) {
                for (int w = 1; w <= Seru_p.W; w++) {
                    max = Math.max(Seru_p.T[o][m][w], max);
                }
            }
        }
        double[][][] worker_T_norm = new double[Seru_p.W + 1][Seru_p.M + 1][Seru_p.O + 1];
        for (int o = 1; o <= Seru_p.O; o++) {
            for (int m = 1; m <= Seru_p.M; m++) {
                for (int w = 1; w <= Seru_p.W; w++) {
                    worker_T_norm[w][m][o] = Seru_p.T[o][m][w]/max;
                }
            }
        }

        return worker_T_norm;
    }
    public static double[][] Get_BaseTime_Norm(Seru.SeruProblem Seru_p){
        double[][] base_norm = new double[Seru_p.M + 1][Seru_p.W + 1];
        // 填充 T 矩阵
        double max = 0;

        for (int m = 1; m <= Seru_p.M; m++) {
            for (int w = 1; w <= Seru_p.W; w++) {
                max = Math.max(Seru_p.baseTime_mi[m][w], max);
            }
        }

        double[][] worker_base_norm = new double[Seru_p.W + 1][Seru_p.M + 1];
        for (int m = 1; m <= Seru_p.M; m++) {
            for (int w = 1; w <= Seru_p.W; w++) {
                worker_base_norm[w][m] = Seru_p.baseTime_mi[m][w]/max;
            }
        }

        return worker_base_norm;
    }



}
