package alog;


import com.fasterxml.jackson.core.JsonProcessingException;
import model.Seru;
import model.Seru.Result;
import model.Seru.SeruProblem;
import model.Seru_output;
import util.DNN_JCompany_Service;
import util.SeruSamplerService;


public class Main_JCompany_con_alpha {
    // ====== 入口参数 ======
    private static final int[] batchSize = {0, 20, 32, 43, 14, 22, 15};
    private static final int W = 6;  // 工人数
    private static final int K = batchSize.length-1;  // 赛汝数
    private static final int M = batchSize.length-1;  // 批次数量
    private static final int O = 4;  // 工序数量

//    private static final int[] batchSize = {0, 2, 3, 4, 1, 2};

    public static void main(String[] args) throws Exception {

        mip_JCompany_con solveOnce = new mip_JCompany_con();

        int Seru_P_num = 30;


        SeruSamplerService.Config cfg = CfgSetting();

        String filepath = "W"+String.valueOf(cfg.W)+"_J"+String.valueOf(cfg.J)+"/";
        String outpath = "C:/code/datasets/Seru_datasets/JCompany/paper2025/instence/"+filepath;


        SeruSamplerService.filecontent F = new SeruSamplerService.filecontent();
        F.GetContent(cfg);
        SeruSamplerService.Result r = SeruSamplerService.GetSamples(F.allBatches, F.sd, F.workerCoeffs, cfg);




        int start = 20;

        int NSF_init = bellNumbers(cfg.W);


        for (int i = start; i <= Seru_P_num; i++){
            int count =0;
            System.out.println("########################进度: " + i+"########################");
//            cfg.AlphaOfScore = 0.5;
            cfg.AlphaOfScoreBatch = 0.05;
            String filename = outpath + i +".xlsx";
            r = SeruSamplerService.GetSamplesFromFile(filename);
            double step = 0.02;
            double[] Alphas = new double[(int) (1/step)];
            double[] gap = new double[(int)(1/step)];
            double[] speedup = new double[(int)(1/step)];
            for (int s = 0; s < (int) (1/step)-1; s++ ) Alphas[s] = (s)*step;
            int j = 0;
            double T = 0;
            Object[][] resultsAll = new Object[(int)(1/step)][20];
            for(int n = 0; n < Alphas.length-2; n++){
                count++;

                cfg.AlphaOfScore = Alphas[n];
                Result[] resultsGNN = GetOneResultAlpha(i,cfg,r,solveOnce);
//                if(n==0) r.SolvingTime = resultsGNN[0].problem.solving_time;
                int flag_e = 0;
                gap[j] = (resultsGNN[0].Cmax - r.cmax)/r.cmax * 100;
                if (gap[j]<=0.000001){
                    flag_e = 1;
                    gap[j] = 0;
                }
                double Gap1 = gap[j];
                double Gap2 = (double) resultsGNN[0].problem.configs.size() /NSF_init;
                double Gap3 = resultsGNN[0].problem.solving_time / r.SolvingTime;
                if (resultsGNN[0].problem.configs.size() == NSF_init){
                    Gap1 = 0;
                    Gap3 = 1;
                    T += resultsGNN[0].problem.solving_time;
                    r.SolvingTime = T/(n+1);
                }
                speedup[j] =(double)r.SolvingTime/resultsGNN[0].problem.solving_time;


                resultsAll[j] = new Object[]{cfg.AlphaOfScore, flag_e, r.cmax, NSF_init, (double)r.SolvingTime/1000,
                        resultsGNN[0].Cmax, Gap1, resultsGNN[0].problem.configs.size(), Gap2, (double)resultsGNN[0].problem.solving_time/1000, Gap3, resultsGNN[0].solvingTimeMIP, resultsGNN[0].solvingTimeGNN,  speedup[j],
                };
                String result_path = outpath + "/result/"+i+"/_" +count+"_"+cfg.AlphaOfScore;
                Seru_output.ExportToExcel_SeruResult_JCompany_Inpaper_con_alpha(result_path, resultsAll);
                j++;
            }
            String outpath2 = outpath + "/result/"+i ;
            Seru_output.ExportToExcel_SeruResult_JCompany_Inpaper_con_alpha(outpath2, resultsAll);
        }

    }

    public static SeruSamplerService.Config CfgSetting(){
        SeruSamplerService.Config cfg = new SeruSamplerService.Config();

        cfg.excelPath ="C:/code/datasets/原/seru_data/pure_seru_data_yu_2014_jd.xlsx";
        cfg.ServerURL = "http://127.0.0.1:5001/run_deepnet";
        cfg.sheetNameBatch = "批次与产品类型关系";
        cfg.sheetNameSkill = "工人与产品类型熟练程度_京东";
        cfg.sheetNameCoeff = "多能工系数";
        cfg.workerMin = 1;  cfg.workerMax = 40;
        cfg.batchMin  = 1;  cfg.batchMax  = 30;
        cfg.W = 7;          cfg.J = 12;
//        cfg.seed = 42L;     // 可选
        cfg.Gap = 0.00;     cfg.StopTime = 300;
        cfg.filepath = "W"+String.valueOf(cfg.W)+"_J"+String.valueOf(cfg.J)+"/";
        cfg.outpath = "C:/code/datasets/Seru_datasets/JCompany/"+cfg.filepath;

        return cfg;
    }

    public static  Result[] GetOneResultAlpha(int i, SeruSamplerService.Config cfg,SeruSamplerService.Result r, mip_JCompany_con solveOnce) throws JsonProcessingException {

        long startTime = 0;
        long endTime = 0;
        // 获取cmax_init
        DNN_JCompany_Service.EdgeResult result = DNN_JCompany_Service.getInitialBound(r, cfg);
        double[][] edge_scores = result.edge_scores;
        double[][] edge_scores_Batch = result.edge_scores_batch;
        double ReasonTime = result.reason_time;
        //            double cmax_init = 2000;
//        r.cmax_init = cmax_init;

        startTime = System.currentTimeMillis();

        SeruProblem Seru_p = Seru.init_JCompany(i, K, O, r, cfg, edge_scores, edge_scores_Batch);


        Result[] results2 = solveOnce.main(Seru_p, false, false, true);
        endTime = System.currentTimeMillis();
        results2[0].problem.solving_time = (endTime - startTime)+(long)(ReasonTime*1000);
        results2[0].solvingTimeMIP = (double) (endTime - startTime)/1000;
        results2[0].solvingTimeGNN = ReasonTime;
        System.out.println("Alpha:"+cfg.AlphaOfScore+",Gap:"+(results2[0].Cmax-r.cmax)/r.cmax+",Cmax=" + results2[0].Cmax+"，总耗时："+ (results2[0].problem.solving_time) + "毫秒，GNN耗时："+ ReasonTime*1000 +"毫秒，求解耗时: " +(endTime - startTime)+
                "毫秒，SeruConfig数："+Seru_p.configs.size()+",conflictPairs:"+Seru_p.conflictPairs.size());

        return results2;

    }

    public static int bellNumbers(int n) {
        int[][] triangle = new int[n + 1][n + 1];
        int[] result = new int[n + 1];

        triangle[0][0] = 1;
        result[0] = 1;

        for (int i = 1; i <= n; i++) {
            triangle[i][0] = triangle[i - 1][i - 1];
            for (int j = 1; j <= i; j++) {
                triangle[i][j] = triangle[i][j - 1] + triangle[i - 1][j - 1];
            }
            result[i] = triangle[i][0];
        }

        return result[n];
    }
}