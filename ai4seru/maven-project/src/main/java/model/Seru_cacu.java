package model;

import java.util.*;

import model.Seru.*;
import org.apache.commons.math3.stat.descriptive.rank.Max;

public class Seru_cacu {

    // ====== 生成T矩阵 ======
    public static double[][][] generateTMatrix(SeruProblem seru_p) {
        double[][][] T = new double[seru_p.O + 1][seru_p.M + 1][seru_p.W + 1];

        // 固定的随机数种子，确保每次生成相同的矩阵
        long seed = 12345;
//        Random rand = new Random(seed);
        Random rand = new Random();

        // 填充 T 矩阵
        for (int o = 1; o <= seru_p.O; o++) {
            for (int m = 1; m <= seru_p.M; m++) {
                for (int w = 1; w <= seru_p.W; w++) {

                    T[o][m][w] = rand.nextDouble() * 10 + 50;
                }
            }
        }

        // 打印矩阵 T 结果
//        System.out.println("Generated T Matrix (工序时间矩阵 T):");
//        for (int o = 1; o <= seru_p.O; o++) {
//            System.out.println("工序 " + o + ":");
//            for (int m = 1; m <= seru_p.M; m++) {
//                System.out.print("批次 " + m + ": ");
//                for (int w = 1; w <= seru_p.W; w++) {
//                    System.out.printf("%.2f ", T[o][m][w]);
//                }
//                System.out.println();
//            }
//            System.out.println();
//        }

        return T;
    }

    // ====== 生成T矩阵JCompany ======
    public static double[][][] generateTMatrix_JCompany(SeruProblem seru_p) {
        double[][][] T = new double[seru_p.O + 1][seru_p.M + 1][seru_p.W + 1];

        // 填充 T 矩阵
        for (int o = 1; o <= seru_p.O; o++) {
            for (int m = 1; m <= seru_p.M; m++) {
                for (int w = 1; w <= seru_p.W; w++) {
                    T[o][m][w] = seru_p.workerProf[w-1][m-1];
                }
            }
        }

        return T;
    }

    // ====== 预计算 baseTime ======baseTime_mi[批次m][工人w]
    public static double[][] precomputeBaseTime(SeruProblem seru_p) {
        double[][] baseTime_mi = new double[seru_p.M + 1][seru_p.W + 1];
        for (int m = 1; m <= seru_p.M; m++) {
            for (int i = 1; i <= seru_p.W; i++) {
                double sum = 0.0;
                for (int o = 1; o <= seru_p.O; o++) {
                    sum += seru_p.T[o][m][i];
                }
                baseTime_mi[m][i] = sum;
            }
        }
        return baseTime_mi;
    }

    // ====== 预计算 tMax ======
    public static double[] precomputeTMax(SeruProblem seru_p) {
        double[] tMax_m = new double[seru_p.M + 1];
        for (int m = 1; m <= seru_p.M; m++) {
            double tmax = 0.0;
            for (int i = 1; i <= seru_p.W; i++) {
                for (int o = 1; o <= seru_p.O; o++) {
                    double v = seru_p.T[o][m][i];
                    if (v > tmax) tmax = v;
                }
            }
            tMax_m[m] = tmax;
        }
        return tMax_m;
    }

    // ====== 根据赛汝构造计算加工时间 p[m][j] ======
    public static double[][] computePFromSeruConfig(SeruConfig cfg, SeruProblem seru_p) {
        int J = cfg.serus.size();
        double[][] p = new double[seru_p.M + 1][J + 1]; // p[m][j], m=0..M, j=1..J

        for (int j = 1; j <= J; j++) {
            Set<Integer> Sj = cfg.serus.get(j - 1); // 该 seru 的工人集合

            for (int m = 1; m <= seru_p.M; m++) {
                double base = 0.0, tmax = 0.0;

                // 计算该seru所有工人的基础时间总和和最大工序时间
                for (int i : Sj) {
                    base += seru_p.baseTime_mi[m][i]; // Σ_o T[o,m,i]

                    // 最大工序时间
                    for (int o = 1; o <= seru_p.O; o++) {
                        double v = seru_p.T[o][m][i];
                        if (v > tmax) tmax = v;
                    }
                }

//                 p[m][j] = base + (batchSize[m] - 1) * tmax
                p[m][j] = base/Sj.size() + (Math.max(1, seru_p.batchSize[m]) - 1) * tmax/Sj.size() ;
            }

            // dummy 行
            p[0][j] = 0.0;
        }

        return p;
    }
    // Processing-time calculation defined in the paper.
    public static double[][] computePFromSeruConfig_V0(SeruConfig cfg, SeruProblem seru_p) {
        int J = cfg.serus.size();
        double[][] p = new double[seru_p.M + 1][J + 1]; // p[m][j], m=0..M, j=1..J

        for (int j = 1; j <= J; j++) {
            Set<Integer> Sj = cfg.serus.get(j - 1); // 该 seru 的工人集合

            for (int m = 1; m <= seru_p.M; m++) {
                double base = 0.0;
                // 计算该seru所有工人的基础时间总和和最大工序时间
                for (int i : Sj) {
                    base += seru_p.baseTime_mi[m][i]; // Σ_o T[o,m,i]

                    // 最大工序时间
                    for (int o = 1; o <= seru_p.O; o++) {
                        double v = seru_p.T[o][m][i];

                    }
                }
                double base_mj = base  * seru_p.batchSize[m]/Sj.size();
                p[m][j] = base_mj/Sj.size() ;
            }

            // dummy 行
            p[0][j] = 0.0;
        }

        return p;
    }

    // ====== 构建 s[m][n][j] ======
    public static double[][][] buildS(int J, int M, int[][] batchInfo) {
        // 简化的切换时间矩阵（所有seru使用相同的切换时间）
        double[][] sBase = {
                {0, 12, 15, 18, 20, 21},
                {12, 0, 9, 13, 14, 21},
                {15, 9, 0, 12, 15, 15},
                {18, 13, 12, 0, 9, 14},
                {20, 14, 15, 9, 0, 11},
                {21, 21, 15, 14, 11, 0},
        };

        int M0 = M + 1;
        double[][][] s = new double[M0][M0][J + 1]; // 1-based with dummy
        for (int j = 1; j <= J; j++) {
            for (int m = 1; m <= M; m++) {
                for (int n = 1; n <= M; n++) {
                    s[m][n][j] = sBase[batchInfo[m-1][1] - 1][batchInfo[n-1][1] - 1];
                }
            }
            // dummy 行/列设为0
            for (int t = 0; t <= M; t++) {
                s[0][t][j] = 0.0;
                s[t][0][j] = 0.0;
            }
        }
        return s;
    }
    // ====== 根据value值和AlphaOfScore生成边的label ======
    public static boolean[][] GetLabel(double[][] value, double AlphaOfScore){
        boolean[][] label = new boolean[value.length][value.length];
        for (int i = 0; i < value.length; i++){
            label[i][i] = false;
            for (int j = i+1; j < value.length; j++){
                if(value[i][j] < AlphaOfScore){
                    label[i][j] = label[j][i] = true;
                }else {
                    label[i][j] = label[j][i] = false;
                }
            }
        }
        return label;
    }
    // ====== 根据labelBatch生成边的conflictPairs ======
    public static List<int[]> GetconflictPairs(boolean[][] label, int M_real){
        List<int[]> conflictPairs = new ArrayList<>();
        for (int i = 1; i <= M_real; i++)
            for (int j = i+1; j <= M_real; j++)
                if (label[i-1][j-1]) conflictPairs.add(new int[]{i, j});
        return conflictPairs;
    }

    // ====== 生成赛汝构造 ======
    public static Set<SeruConfig> generateSeruConfigs(int W, int K) {
        Set<SeruConfig> configs = new HashSet<>();
        List<Integer> workers = new ArrayList<>();
        for (int i = 1; i <= W; i++) workers.add(i);

//        generateConfigsRecursive(workers, K, new ArrayList<>(), configs);
//        configs = generateExact(W, K);
        configs = generateAll(W);
        return configs;
    }

    public static void generateConfigsRecursive(List<Integer> remainingWorkers, int remainingSerus,
                                                 List<Set<Integer>> currentSerus, Set<SeruConfig> configs) {
        if (remainingSerus == 0) {
            if (remainingWorkers.isEmpty()) {
                configs.add(new SeruConfig(new ArrayList<>(currentSerus)));
            }
            return;
        }

        if (remainingWorkers.isEmpty()) return;

        // 枚举第一个seru可能包含的工人组合
        int n = remainingWorkers.size();
        for (int mask = 1; mask < (1 << n); mask++) {
            Set<Integer> seru = new HashSet<>();
            List<Integer> remaining = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) != 0) {
                    seru.add(remainingWorkers.get(i));
                } else {
                    remaining.add(remainingWorkers.get(i));
                }
            }

            currentSerus.add(seru);
            generateConfigsRecursive(remaining, remainingSerus - 1, currentSerus, configs);
            currentSerus.remove(currentSerus.size() - 1);
        }
    }

    /** 生成 {1..W} 的所有划分（无序、去重）。 */
    public static Set<SeruConfig> generateAll(int W) {
        if (W <= 0) return Collections.emptySet();

        // base: {{1}}
        Set<SeruConfig> level = new LinkedHashSet<>();
        List<Set<Integer>> base = new ArrayList<>();
        Set<Integer> s1 = new HashSet<>();
        s1.add(1);
        base.add(s1);
        level.add(new SeruConfig(normalize(base)));

        // 递推：从 w=2..W
        for (int w = 2; w <= W; w++) {
            Set<SeruConfig> next = new LinkedHashSet<>();
            for (SeruConfig prev : level) {
                // 方式1：新工人 w 独立成 seru
                {
                    List<Set<Integer>> parts = deepCopy(prev.serus);
                    Set<Integer> nw = new HashSet<>();
                    nw.add(w);
                    parts.add(nw);
                    next.add(new SeruConfig(normalize(parts)));
                }
                // 方式2：新工人 w 加入每个现有 seru
                for (int i = 0; i < prev.serus.size(); i++) {
                    List<Set<Integer>> parts = deepCopy(prev.serus);
                    parts.get(i).add(w);
                    next.add(new SeruConfig(normalize(parts)));
                }
            }
            level = next;
        }
        return level;
    }

    /** 生成 {1..W} 划分为恰好 K 个 seru 的所有配置。 */
    public static Set<SeruConfig> generateExact(int W, int K) {
        if (W <= 0 || K <= 0 || K > W) return Collections.emptySet();

        // base: {{1}} 仅当 K>=1
        Set<SeruConfig> level = new LinkedHashSet<>();
        List<Set<Integer>> base = new ArrayList<>();
        Set<Integer> s1 = new HashSet<>();
        s1.add(1);
        base.add(s1);
        level.add(new SeruConfig(normalize(base)));

        for (int w = 2; w <= W; w++) {
            Set<SeruConfig> next = new LinkedHashSet<>();
            for (SeruConfig prev : level) {
                int s = prev.serus.size();

                // 允许新开 seru 的前提：s+1 <= K
                if (s + 1 <= K) {
                    List<Set<Integer>> parts = deepCopy(prev.serus);
                    Set<Integer> nw = new HashSet<>();
                    nw.add(w);
                    parts.add(nw);
                    next.add(new SeruConfig(normalize(parts)));
                }

                // 将 w 加入任一已有 seru
                for (int i = 0; i < s; i++) {
                    List<Set<Integer>> parts = deepCopy(prev.serus);
                    parts.get(i).add(w);
                    next.add(new SeruConfig(normalize(parts)));
                }
            }
            level = next;
        }

        // 过滤恰好 K 个 seru
        Set<SeruConfig> out = new LinkedHashSet<>();
        for (SeruConfig c : level) if (c.serus.size() == K) out.add(c);
        return out;
    }

    // ========= 工具函数 =========

    /** 深拷贝 List<Set<Integer>>。 */
    private static List<Set<Integer>> deepCopy(List<Set<Integer>> src) {
        List<Set<Integer>> copy = new ArrayList<>(src.size());
        for (Set<Integer> s : src) copy.add(new HashSet<>(s));
        return copy;
    }

    /** 规范化 seru 列表顺序：按 (最小元素, 大小, 词典序) 排序，保证同构划分有唯一表示。 */
    private static List<Set<Integer>> normalize(List<Set<Integer>> parts) {
        // 将每个集合转为 TreeSet 便于比较
        List<TreeSet<Integer>> canon = new ArrayList<>(parts.size());
        for (Set<Integer> s : parts) canon.add(new TreeSet<>(s));

        canon.sort((a, b) -> {
            // 按最小元素排序（集合非空）
            int mina = a.first();
            int minb = b.first();
            if (mina != minb) return Integer.compare(mina, minb);
            // 再按大小
            if (a.size() != b.size()) return Integer.compare(a.size(), b.size());
            // 最后按词典序
            Iterator<Integer> ia = a.iterator(), ib = b.iterator();
            while (ia.hasNext() && ib.hasNext()) {
                int cmp = Integer.compare(ia.next(), ib.next());
                if (cmp != 0) return cmp;
            }
            return 0;
        });

        // 返回为 List<Set<Integer>>（SeruConfig 构造中会拷贝为 HashSet）
        List<Set<Integer>> out = new ArrayList<>(canon.size());
        out.addAll(canon);
        return out;
    }
    /** 计算 label value矩阵 */
    public static Result cacu_LabelandValue(Result[] results){
        double[][] value = new double [results[0].problem.W][results[0].problem.W];
        double[][] label = new double [results[0].problem.W][results[0].problem.W];
        double max = 0;
        double min = 1e8;
        double[][] values = new double [results[0].problem.W][results[0].problem.W];
        double[][] labels = new double [results[0].problem.W][results[0].problem.W];
        double[][] counts = new double [results[0].problem.W][results[0].problem.W];

        for(Result r:results){
            
            for(int i = 0; i <r.config.serus.size(); i++){
                Object[] n = r.config.serus.get(i).toArray();

                for(int j1 = 0; j1 < n.length-1; j1++){
                    for(int j2 = j1+1; j2 < n.length; j2++){
                        values[(Integer)n[j1]-1][(Integer)n[j2]-1] = values[(Integer)n[j2]-1][(Integer)n[j1]-1] += r.Cmax;
                        labels[(Integer)n[j1]-1][(Integer)n[j2]-1] = labels[(Integer)n[j2]-1][(Integer)n[j1]-1] += Math.floor(r.value);
                        counts[(Integer)n[j1]-1][(Integer)n[j2]-1] = counts[(Integer)n[j2]-1][(Integer)n[j1]-1] += 1;
                    }
                }
            }
        }

        for(int i = 0; i < value.length; i++){
            for (int j = i+1; j < value.length; j++){
                value[i][j] = value[j][i] = (values[i][j]/counts[i][j]);
                label[i][j] = label[j][i] = Math.min(1,labels[i][j]);
                max = Math.max(max,value[i][j]);
                min = Math.min(min,value[i][j]);
            }
        }

        for(int i = 0; i < value.length; i++){
            for (int j = i+1; j < value.length; j++){
                double v = (value[i][j]-max)/(min-max);
                value[i][j] = value[j][i] = Math.max(v, label[i][j]);
            }
        }

        results[0].problem.value = value;
        results[0].problem.label = label;
        return results[0];
    }

    public static Set<SeruConfig> generateSeruConfigs(int W, int K, boolean[][] forbiddenPairs) {
        return generateAllWithConstraints(W, forbiddenPairs);
    }
    /** 生成 {1..W} 的所有划分（无序、去重），并满足 forbiddenPairs 约束。 */
    public static Set<SeruConfig> generateAllWithConstraints(int W, boolean[][] conflict) {
        if (W <= 0) return Collections.emptySet();

        // base: {{1}}
        Set<SeruConfig> level = new LinkedHashSet<>();
        List<Set<Integer>> base = new ArrayList<>();
        Set<Integer> s1 = new HashSet<>();
        s1.add(1);
        base.add(s1);
        level.add(new SeruConfig(normalize(base)));

        // 递推：从 w=2..W
        for (int w = 2; w <= W; w++) {
            Set<SeruConfig> next = new LinkedHashSet<>();
            for (SeruConfig prev : level) {
                // 方式1：新工人 w 独立成 seru（一定合法，因为单人组不会违反“成对冲突”）
                {
                    List<Set<Integer>> parts = deepCopy(prev.serus);
                    Set<Integer> nw = new HashSet<>();
                    nw.add(w);
                    parts.add(nw);
                    next.add(new SeruConfig(normalize(parts)));
                }
                // 方式2：新工人 w 加入每个现有 seru（加入前先检查是否与组内任意人冲突）
                for (int i = 0; i < prev.serus.size(); i++) {
                    Set<Integer> targetSeru = prev.serus.get(i);
                    if (!canJoin(targetSeru, w, conflict)) {
                        // 这个 seru 里有和 w 冲突的工人，跳过这种扩展
                        continue;
                    }
                    List<Set<Integer>> parts = deepCopy(prev.serus);
                    parts.get(i).add(w);
                    next.add(new SeruConfig(normalize(parts)));
                }
            }
            level = next;
        }
        return level;
    }
    /** 把二维数组 {i,j} 转成对称的冲突矩阵 conflict[a][b] = true 表示 a、b 不能同组 */
    private static boolean[][] buildConflictMatrix(int W, int[][] forbiddenPairs) {
        boolean[][] conflict = new boolean[W + 1][W + 1]; // 1..W 使用

        if (forbiddenPairs == null) return conflict;

        for (int[] p : forbiddenPairs) {
            if (p == null || p.length < 2) continue;
            int a = p[0];
            int b = p[1];
            if (a == b) continue; // 自己和自己冲突没意义
            // 只接受合法工号
            if (1 <= a && a <= W && 1 <= b && b <= W) {
                conflict[a][b] = true;
                conflict[b][a] = true;
            }
        }
        return conflict;
    }

    /** 判断工人 w 加入某个 seru 是否违反冲突约束 */
    private static boolean canJoin(Set<Integer> seru, int w, boolean[][] conflict) {
        for (int worker : seru) {
//            System.out.println(worker);
            if (conflict[worker-1][w-1]) {
                return false;
            }
        }
        return true;
    }


}
