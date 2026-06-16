package alog;


import model.Seru;
import model.Seru.*;
import model.Seru_output;
import model.Seru_output.*;

public class Main {
    // ====== 入口参数 ======
    private static final int[] batchSize = {0, 20, 32, 43, 14, 22, 15};
    private static final int W = 6;  // 工人数
    private static final int K = batchSize.length-1;  // 赛汝数
    private static final int M = batchSize.length-1;  // 批次数量
    private static final int O = 4;  // 工序数量

//    private static final int[] batchSize = {0, 2, 3, 4, 1, 2};

    public static void main(String[] args){

        mip solveOnce = new mip();

        String outpath = "C:/code/datasets/Seru_datasets/randomT50-60_W2_P_V0/";
        int Seru_P_num = 10000;

        for (int i = 9819; i <= Seru_P_num; i++){
            System.out.println("进度: " + i);
            SeruProblem Seru_p = Seru.init(i, W, K, M, O, batchSize);
//            Result[] results = solveOnce.main(Seru_p, true);
            Result[] results2 = solveOnce.main(Seru_p, false);
//            Result best_r = results[0];
//            Result best_r2 = results2[0];
            for (Result r :results2){
                Seru_output.ExportToExcel_SeruResult(outpath, i, r);
//                Seru_output.printSolution(r);
            }
//            System.out.println("#######S=true");
//            Seru_output.printBestSolution(best_r);
//            System.out.println("#######S=false");
//            Seru_output.printBestSolution(best_r2);

//            double[][][] T_norm = Seru_output.Get_T_Norm(Seru_p);
//            double[][] baseTime_norm = Seru_output.Get_BaseTime_Norm(Seru_p);
        }

    }

}