package alog;


import com.fasterxml.jackson.core.JsonProcessingException;
import model.Seru;
import model.Seru.Result;
import model.Seru.SeruProblem;
import model.Seru_output;
import util.DNN_JCompany_Service;
import util.SeruSamplerService;

public class Main_JCompany_con {
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
        String outpath = "C:/code/datasets/Seru_datasets/JCompany/paper2025/"+filepath;


        SeruSamplerService.filecontent F = new SeruSamplerService.filecontent();
        F.GetContent(cfg);
        SeruSamplerService.Result r = SeruSamplerService.GetSamples(F.allBatches, F.sd, F.workerCoeffs, cfg);

        long last_solving_time = 0;
        double timeMIP = 0;
        double timeGNN = 0;

        double[] gap = new double[Seru_P_num];
        double[] speedup = new double[Seru_P_num];
        double[] times = new double[Seru_P_num];
        double[] gap2 = new double[Seru_P_num];
        double[] speedup2 = new double[Seru_P_num];
        double[] times2 = new double[Seru_P_num];
        Object[][] resultsAll = new Object[Seru_P_num+2][20];
        int Exact_count_1 = 0;
        int Exact_count_2 = 0;
        int start = 1;
        int count =0;

        for (int i = start; i <= Seru_P_num; i++){
            count += 1;
            System.out.println("########################进度: " + i+"########################");
            cfg.AlphaOfScore = 0.5;
            cfg.AlphaOfScoreBatch = 0.05;
            r = SeruSamplerService.GetSamples(F.allBatches, F.sd, F.workerCoeffs,cfg);
            Result[] resultsGNN = GetOneResultGNN(i,cfg,r,solveOnce,last_solving_time);
            Result[] resultsGNN2 = GetOneResultGNN2(i,cfg,r,solveOnce,last_solving_time);
            Result[] resultsMIP = GetOneResultMIP(i,cfg,r,solveOnce,last_solving_time);


            gap[i-1] = (resultsGNN[0].Cmax - resultsMIP[0].Cmax)/resultsMIP[0].Cmax;
            times[i-1] = (double) (resultsMIP[0].problem.solving_time - resultsGNN[0].problem.solving_time)/resultsMIP[0].problem.solving_time;
            speedup[i-1] =(double)resultsMIP[0].problem.solving_time/resultsGNN[0].problem.solving_time;
            double gap_sum = 0;
            for (double g : gap) gap_sum+=g;
            double gap_ave  =gap_sum/count;
            double t_sum = 0;
            for (double t : times) t_sum+=t;
            double t_ave  =t_sum/count;
            if (gap[i-1]<=0.0000001) {
                Exact_count_1 += 1;
                gap[i-1] = 0;
            }
            double e = (double) Exact_count_1/count;
            System.out.println("GNN: 平均gap："+ gap_ave*100 +"%，平均效率提升："+t_ave*100+"%, 求得精确解率："+e*100+"%");

            gap2[i-1] = (resultsGNN2[0].Cmax - resultsMIP[0].Cmax)/resultsMIP[0].Cmax;
            times2[i-1] = (double) (resultsMIP[0].problem.solving_time - resultsGNN2[0].problem.solving_time)/resultsMIP[0].problem.solving_time;
            speedup2[i-1] =(double)resultsMIP[0].problem.solving_time/resultsGNN2[0].problem.solving_time;
            double gap_sum2 = 0;
            for (double g : gap2) gap_sum2+=g;
            double gap_ave2  =gap_sum2/count;
            double t_sum2 = 0;
            for (double t : times2) t_sum2+=t;
            double t_ave2  =t_sum2/count;
            if (gap2[i-1]<=0.0000001) {
                Exact_count_2 += 1;
                gap2[i-1] = 0;
            }
            double e2 = (double) Exact_count_2 /count;
            System.out.println("GNN2: 平均gap："+ gap_ave2*100 +"%，平均效率提升："+t_ave2*100+"%, 求得精确解率："+e2*100+"%");
            String file_path = outpath + "/";
            Seru_output.ExportToExcel_SeruResult_JCompany_Inpaper(file_path, i, resultsMIP, r);
            int flag_e = 0;
            String Gap1 = String.valueOf(gap[i-1])+"%";
            String Gap2 = String.valueOf(gap2[i-1])+"%";
            if (gap2[i-1]<=0.000001) flag_e = 1;
            resultsAll[i-1] = new Object[]{i, flag_e, resultsMIP[0].Cmax, resultsMIP[0].problem.configs.size(), (double)resultsMIP[0].problem.solving_time/1000,
                    resultsGNN[0].Cmax, resultsGNN[0].problem.configs.size(), (double)resultsGNN[0].problem.solving_time/1000, resultsGNN[0].solvingTimeMIP, resultsGNN[0].solvingTimeGNN, Gap1, speedup[i-1],
                    resultsGNN2[0].Cmax, resultsGNN2[0].problem.configs.size(), (double)resultsGNN2[0].problem.solving_time/1000, resultsGNN2[0].solvingTimeMIP, resultsGNN2[0].solvingTimeGNN, Gap2, speedup2[i-1],resultsGNN2[0].problem.conflictPairs.size()};
            String result_path = outpath + "/_" +count+"_"+i;
            Seru_output.ExportToExcel_SeruResult_JCompany_Inpaper_con(result_path, resultsAll);
        }

        double gap_sum = 0;
        for (double g : gap) gap_sum+=g;
        double gap_ave  =gap_sum/Seru_P_num;
        double t_sum = 0;
        for (double t : times2) t_sum+=t;
        double t_ave  =t_sum/Seru_P_num;
        double speed_sum = 1;
        for (double s: speedup) speed_sum *=s;
        double s_ave = Math.pow(speed_sum, (double) 1 /Seru_P_num);

        double gap_sum2 = 0;
        for (double g : gap2) gap_sum2+=g;
        double gap_ave2  =gap_sum2/Seru_P_num;
        double t_sum2 = 0;
        for (double t : times2) t_sum2+=t;
        double t_ave2  =t_sum2/Seru_P_num;
        double speed_sum2 = 1;
        for (double s: speedup2) speed_sum2 *=s;
        double s_ave2 = Math.pow(speed_sum2,(double) 1/Seru_P_num);
        resultsAll[Seru_P_num] = new Object[]{"GNN: 平均gap："+ gap_ave*100 +"%，平均效率提升："+t_ave*100+"%, 求得精确解数量："+Exact_count_1+",几何加速比："+s_ave};
        resultsAll[Seru_P_num+1] = new Object[]{"GNN2: 平均gap："+ gap_ave2*100 +"%，平均效率提升："+t_ave2*100+"%, 求得精确解数量："+Exact_count_2+",几何加速比："+s_ave2};
        Seru_output.ExportToExcel_SeruResult_JCompany_Inpaper_con(outpath, resultsAll);


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
        cfg.W = 8;          cfg.J = 10;
//        cfg.seed = 42L;     // 可选
        cfg.Gap = 0.00;     cfg.StopTime = 300;
        cfg.filepath = "W"+String.valueOf(cfg.W)+"_J"+String.valueOf(cfg.J)+"/";
        cfg.outpath = "C:/code/datasets/Seru_datasets/JCompany/"+cfg.filepath;

        return cfg;
    }

    public static  Result[] GetOneResultGNN(int i, SeruSamplerService.Config cfg,SeruSamplerService.Result r, mip_JCompany_con solveOnce, long last_solving_time) throws JsonProcessingException {

        long startTime = 0;
        long endTime = 0;
        // 获取cmax_init
        DNN_JCompany_Service.EdgeResult result = DNN_JCompany_Service.getInitialBound(r, cfg);
        double[][] edge_scores = result.edge_scores;
        double ReasonTime = result.reason_time;
        //            double cmax_init = 2000;
//        r.cmax_init = cmax_init;

        startTime = System.currentTimeMillis();

        SeruProblem Seru_p = Seru.init_JCompany(i, K, O, r, cfg, edge_scores);


        Result[] results2 = solveOnce.main(Seru_p, false, false, false);
        endTime = System.currentTimeMillis();
        results2[0].problem.solving_time = (endTime - startTime)+(long)(ReasonTime*1000);
        results2[0].solvingTimeMIP = (double) (endTime - startTime)/1000;
        results2[0].solvingTimeGNN = ReasonTime;
        System.out.println("GNN:Cmax=" + results2[0].Cmax+"，总耗时："+ (results2[0].problem.solving_time) + "毫秒,GNN耗时："+ ReasonTime*1000 +"毫秒，求解耗时: " +(endTime - startTime)+"毫秒,SeruConfig数："+Seru_p.configs.size());

        return results2;

    }
    // config+schedule
    public static  Result[] GetOneResultGNN2(int i, SeruSamplerService.Config cfg,SeruSamplerService.Result r, mip_JCompany_con solveOnce, long last_solving_time) throws JsonProcessingException {

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
        System.out.println("GNN2:Cmax=" + results2[0].Cmax+"，总耗时："+ (results2[0].problem.solving_time) + "毫秒，GNN耗时："+ ReasonTime*1000 +"毫秒，求解耗时: " +(endTime - startTime)+
                "毫秒，SeruConfig数："+Seru_p.configs.size()+",conflictPairs:"+Seru_p.conflictPairs.size());

        return results2;

    }
    public static  Result[] GetOneResultMIP(int i, SeruSamplerService.Config cfg,SeruSamplerService.Result r, mip_JCompany_con solveOnce, long last_solving_time) throws JsonProcessingException {

        long startTime = 0;
        long endTime = 0;
        // 获取cmax_init
//        DNN_JCompany_Service.EdgeResult result = DNN_JCompany_Service.getInitialBound(r, cfg);
//        double[][] edge_scores = result.edge_scores;
//        double ReasonTime = result.reason_time;
        //            double cmax_init = 2000;
//        r.cmax_init = cmax_init;

        startTime = System.currentTimeMillis();

        SeruProblem Seru_p = Seru.init_JCompany(i, K, O, r, cfg);


        Result[] results2 = solveOnce.main(Seru_p, false, false, false);
        endTime = System.currentTimeMillis();
        results2[0].problem.solving_time = endTime - startTime;
        System.out.println("MIP:Cmax=" + results2[0].Cmax+"，求解耗时: " + ((results2[0].problem.solving_time))+ "毫秒，SeruConfig数："+Seru_p.configs.size());

        return results2;

    }
}