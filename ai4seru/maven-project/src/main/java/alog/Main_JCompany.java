package alog;


import model.Seru;
import model.Seru.Result;
import model.Seru.SeruProblem;
import model.Seru_cacu;
import model.Seru_output;

import util.CceaBoundService;
import util.SeruSamplerService;

public class Main_JCompany {
    // ====== 入口参数 ======
    private static final int[] batchSize = {0, 20, 32, 43, 14, 22, 15};
    private static final int W = 6;  // 工人数
    private static final int K = batchSize.length-1;  // 赛汝数
    private static final int M = batchSize.length-1;  // 批次数量
    private static final int O = 4;  // 工序数量

//    private static final int[] batchSize = {0, 2, 3, 4, 1, 2};

    public static void main(String[] args) throws Exception {

        mip_JCompany solveOnce = new mip_JCompany();

        long startTime = 0;
        long endTime = 0;


        int Seru_P_num = 5000;

        SeruSamplerService.Config cfg = new SeruSamplerService.Config();

        cfg.excelPath ="C:/code/datasets/原/seru_data/pure_seru_data_yu_2014_jd.xlsx";
        cfg.ServerURL = "http://127.0.0.1:5000/run_ccea";
        cfg.sheetNameBatch = "批次与产品类型关系";
        cfg.sheetNameSkill = "工人与产品类型熟练程度_京东";
        cfg.sheetNameCoeff = "多能工系数";
        cfg.workerMin = 1;  cfg.workerMax = 40;
        cfg.batchMin  = 1;  cfg.batchMax  = 30;
        cfg.W = 8;          cfg.J = 10;
//        cfg.seed = 42L;     // 可选
        cfg.Gap = 0.01;     cfg.StopTime = 300;
        String filepath = "W"+String.valueOf(cfg.W)+"_J"+String.valueOf(cfg.J)+"/";
        String outpath = "C:/code/datasets/Seru_datasets/JCompany/"+filepath;
        SeruSamplerService.filecontent F = new SeruSamplerService.filecontent();
        F.GetContent(cfg);

        SeruSamplerService.Result r = SeruSamplerService.GetSamples(F.allBatches, F.sd, F.workerCoeffs, cfg);


        int[][] batchInfo = r.batchTable();
        Object[][] workerInfo = r.workerTable();

        java.util.List<SeruSamplerService.BatchInfo> batchList = r.batches;  // 每项含 batchId/productType/batchSize
        int[] workerIds = r.workerIds;                                       // 长度 W
        double[][] prof = r.workerProficiencies;                             // [W,J] 与 r.workerColumnNames 对齐
        long last_solving_time = 0;


        for (int i = 1; i <= Seru_P_num; i++){
            System.out.println("进度: " + i);
            r = SeruSamplerService.GetSamples(F.allBatches, F.sd, F.workerCoeffs,cfg);
            // 获取cmax_init
            double cmax_init = CceaBoundService.getInitialBound(r, cfg);
//            double cmax_init = 2000;
            r.cmax_init = cmax_init;

            startTime = System.currentTimeMillis();
            SeruProblem Seru_p = Seru.init_JCompany(i, K, O, r, cfg);


            Result[] results2 = solveOnce.main(Seru_p, false);
            endTime = System.currentTimeMillis();
            results2[0].problem.solving_time = endTime - startTime;
            System.out.println("进度:" + i+ "，执行耗时: " + (results2[0].problem.solving_time)/1000 + "秒,Cmax="+results2[0].Cmax);
            if (last_solving_time == 0 || (results2[0].problem.solving_time < (last_solving_time*5))) {
                Object[][] RofConfs = new Object[results2.length][5];
                for (int j = 0; j <results2.length; j++){
                    RofConfs[j][0] = results2[j].idx;
                    RofConfs[j][1] = results2[j].value;
                    RofConfs[j][2] = results2[j].Cmax;
                    RofConfs[j][3] = results2[j].config;
                    RofConfs[j][4] = results2[j].batchToSeru;
                }
                results2[0].problem.ResultofConfigs = RofConfs;
                results2[0] = Seru_cacu.cacu_LabelandValue(results2);
                int index = i/100;
                String file_path = outpath+String.valueOf(index) + "/";
                Seru_output.ExportToExcel_SeruResult_JCompany(file_path, i, results2);
                last_solving_time = results2[0].problem.solving_time;
            }else {
                i--;
            }



//            System.out.println("****************************");
//            System.out.println("****************************");
//            for (Result r :results2){
//                Seru_output.ExportToExcel_SeruResult_JCompany(outpath, i, r);
////                Seru_output.printSolution(r);
//            }
//            System.out.println("#######S=true");
//            Seru_output.printBestSolution(best_r);
//            System.out.println("#######S=false");
//            Seru_output.printBestSolution(best_r2);

//            double[][][] T_norm = Seru_output.Get_T_Norm(Seru_p);
//            double[][] baseTime_norm = Seru_output.Get_BaseTime_Norm(Seru_p);
        }

    }

}