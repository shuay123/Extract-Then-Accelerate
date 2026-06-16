package model;

import java.util.*;
import util.ExcelMultiSheetWriter;
import util.SeruSamplerService;
import java.io.IOException;

public class Seru {
    public static class SeruProblem {
        public int W;  // 工人数
        public int K;  // 赛汝数
        public int M;  // 批次数量
        public int O;  // 工序数量
        public int index;
        public double CmaxInit;
        public double Gap;
        public double StopTime;

        public int[] batchSize;
        public int[] batchIds;
        public int[][] batchInfo;
        public int[] workerIds;
        public double[][] workerProf;

        public double[][][] T;
        public double[][] baseTime_mi;
        public double[] tMax_m;
        public double[][] sgiven;
        public Set<SeruConfig> configs;
        public Object[][] ResultofConfigs;


        public long solving_time;
        public double[][] label;
        public boolean[][] label2; //阈值筛选后的工人标签
        public boolean[][] label_batch; //阈值筛选后的批次标签
        public List<int[]> conflictPairs;
        public double[][] value;


    }
    // ====== Seru问题初始化 ======
    public static SeruProblem init(int index, int w, int k, int m, int o, int[] BatchSize){
        SeruProblem seru_p = new SeruProblem();
        seru_p.W = w;
        seru_p.K = k;
        seru_p.M = m;
        seru_p.O = o;
        seru_p.index = index;

        seru_p.batchSize = BatchSize;
        seru_p.T = Seru_cacu.generateTMatrix(seru_p);
        seru_p.baseTime_mi = Seru_cacu.precomputeBaseTime(seru_p);
        seru_p.tMax_m = Seru_cacu.precomputeTMax(seru_p);
        seru_p.configs = Seru_cacu.generateSeruConfigs(seru_p.W, seru_p.K);

        seru_p.solving_time = 0;

        try {
            if (seru_p.configs.isEmpty()) {
                System.out.println("没有生成任何赛汝构造。");
            }
//            System.out.println("构造数量: " + seru_p.configs.size());
        } catch (Exception e) {
            System.err.println("出现异常: " + e.getMessage());
            e.printStackTrace();
        }

        return seru_p;
    }

    // ====== Seru问题初始化 ======
    public static SeruProblem init_JCompany(int index, int k, int o, SeruSamplerService.Result r, SeruSamplerService.Config cfg){
        SeruProblem seru_p = new SeruProblem();
        seru_p.W = r.workerIds.length;
        seru_p.K = k;
        seru_p.M = r.batchIds.length;
        seru_p.O = o;
        seru_p.index = index;

        seru_p.batchIds = r.batchIds;
        seru_p.batchSize = r.batchSize;
        seru_p.batchInfo = r.batchTable();
        seru_p.workerIds = r.workerIds;
        seru_p.workerProf = r.workerProficiencies;
        seru_p.CmaxInit = r.cmax_init;
        seru_p.Gap = r.Gap;
        seru_p.StopTime = cfg.StopTime;

        seru_p.T = Seru_cacu.generateTMatrix_JCompany(seru_p); // T[o][m][w]：批次m中的工序o由工人w单独完成所需时间
        seru_p.baseTime_mi = Seru_cacu.precomputeBaseTime(seru_p); // baseTime_mi工人i单独完成批次m中一个工件所有工序的时间
        seru_p.tMax_m = Seru_cacu.precomputeTMax(seru_p);

        seru_p.configs = Seru_cacu.generateSeruConfigs(seru_p.W, seru_p.K);

        seru_p.solving_time = 0;

        try {
            if (seru_p.configs.isEmpty()) {
                System.out.println("没有生成任何赛汝构造。");
            }
//            System.out.println("构造数量: " + seru_p.configs.size());
        } catch (Exception e) {
            System.err.println("出现异常: " + e.getMessage());
            e.printStackTrace();
        }

        return seru_p;
    }
    public static SeruProblem init_JCompany(int index, int k, int o, SeruSamplerService.Result r, SeruSamplerService.Config cfg, double[][] edge_scores){
        SeruProblem seru_p = new SeruProblem();
        seru_p.W = r.workerIds.length;
        seru_p.K = k;
        seru_p.M = r.batchIds.length;
        seru_p.O = o;
        seru_p.index = index;

        seru_p.batchIds = r.batchIds;
        seru_p.batchSize = r.batchSize;
        seru_p.batchInfo = r.batchTable();
        seru_p.workerIds = r.workerIds;
        seru_p.workerProf = r.workerProficiencies;
        seru_p.CmaxInit = r.cmax_init;
        seru_p.Gap = r.Gap;
        seru_p.StopTime = cfg.StopTime;

        seru_p.T = Seru_cacu.generateTMatrix_JCompany(seru_p); // T[o][m][w]：批次m中的工序o由工人w单独完成所需时间
        seru_p.baseTime_mi = Seru_cacu.precomputeBaseTime(seru_p); // baseTime_mi工人i单独完成批次m中一个工件所有工序的时间
        seru_p.tMax_m = Seru_cacu.precomputeTMax(seru_p);



        seru_p.solving_time = 0;
        seru_p.value = edge_scores;
        seru_p.label2 = Seru_cacu.GetLabel(edge_scores, cfg.AlphaOfScore);
        seru_p.configs = Seru_cacu.generateSeruConfigs(seru_p.W, seru_p.K, seru_p.label2);

        try {
            if (seru_p.configs.isEmpty()) {
                System.out.println("没有生成任何赛汝构造。");
            }
//            System.out.println("构造数量: " + seru_p.configs.size());
        } catch (Exception e) {
            System.err.println("出现异常: " + e.getMessage());
            e.printStackTrace();
        }

        return seru_p;
    }
    public static SeruProblem init_JCompany(int index, int k, int o, SeruSamplerService.Result r, SeruSamplerService.Config cfg, double[][] edge_scores, double[][] edge_scoresBatch){
        SeruProblem seru_p = new SeruProblem();
        seru_p.W = r.workerIds.length;
        seru_p.K = k;
        seru_p.M = r.batchIds.length;
        seru_p.O = o;
        seru_p.index = index;

        seru_p.batchIds = r.batchIds;
        seru_p.batchSize = r.batchSize;
        seru_p.batchInfo = r.batchTable();
        seru_p.workerIds = r.workerIds;
        seru_p.workerProf = r.workerProficiencies;
        seru_p.CmaxInit = r.cmax_init;
        seru_p.Gap = r.Gap;
        seru_p.StopTime = cfg.StopTime;

        seru_p.T = Seru_cacu.generateTMatrix_JCompany(seru_p); // T[o][m][w]：批次m中的工序o由工人w单独完成所需时间
        seru_p.baseTime_mi = Seru_cacu.precomputeBaseTime(seru_p); // baseTime_mi工人i单独完成批次m中一个工件所有工序的时间
        seru_p.tMax_m = Seru_cacu.precomputeTMax(seru_p);



        seru_p.solving_time = 0;
        seru_p.value = edge_scores;
        seru_p.label2 = Seru_cacu.GetLabel(edge_scores, cfg.AlphaOfScore);
        seru_p.label_batch = Seru_cacu.GetLabel(edge_scoresBatch, cfg.AlphaOfScoreBatch); //GetconflictPairs
        seru_p.conflictPairs = Seru_cacu.GetconflictPairs(seru_p.label_batch, seru_p.M);
        seru_p.configs = Seru_cacu.generateSeruConfigs(seru_p.W, seru_p.K, seru_p.label2);

        try {
            if (seru_p.configs.isEmpty()) {
                System.out.println("没有生成任何赛汝构造。");
            }
//            System.out.println("构造数量: " + seru_p.configs.size());
        } catch (Exception e) {
            System.err.println("出现异常: " + e.getMessage());
            e.printStackTrace();
        }

        return seru_p;
    }

    // ====== 数据结构定义 ======
    public static class SeruConfig {
        public List<Set<Integer>> serus; // 每个seru包含的工人集合

        public SeruConfig(List<Set<Integer>> serus) {
            this.serus = new ArrayList<>();
            for (Set<Integer> seru : serus) {
                this.serus.add(new HashSet<>(seru));
            }
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            SeruConfig that = (SeruConfig) obj;
            return Objects.equals(serus, that.serus);
        }

        @Override
        public int hashCode() {
            return Objects.hash(serus);
        }

        @Override
        public String toString() {
            return "SeruConfig{serus=" + serus + "}";
        }
    }

    public static class Result {
        public  SeruProblem problem;
        public SeruConfig config;
        public double Cmax;
        public int[] batchToSeru; // 1..M
        public int J, M;
        public int idx;
        public double value;
        public double solvingTimeMIP;
        public double solvingTimeGNN;
    }


}
