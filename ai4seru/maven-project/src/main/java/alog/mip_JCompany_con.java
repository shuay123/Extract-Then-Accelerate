package alog;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;
import ilog.cplex.IloCplex.Param;
import model.Seru.Result;
import model.Seru.SeruConfig;
import model.Seru.SeruProblem;
import model.Seru_cacu;

import java.util.ArrayList;
import java.util.List;

/**
 * MIP model:
 * - 使用LBBD8中的加工时间计算方式，基于T矩阵和批次大小
 * - 枚举所有固定赛汝数 K 的赛汝构造；
 * - 对每个构造，根据T矩阵计算该构造下的 p[m][j]（加工时间）；
 * - 代入 MIP (22)-(29) 求解 Cmax；
 * - 取最小的 Cmax 及其对应构造与分配。
 */
public class mip_JCompany_con {

    // ====== 程序入口 ======
    public static Result[] main(SeruProblem Seru_p, boolean s_flag, boolean print_flag, boolean B_s_flag) {
        try {
            // 3) 枚举构造，分别求解
            Result best = null;
            int idx = 1;
            long startTime = 0;
            long endTime = 0;

            double WorstCmax = 0;
            Result[] results = new Result[Seru_p.configs.size()+1];
            List<Result> results_best = new ArrayList<Result>();
            for (SeruConfig cfg : Seru_p.configs) {

                startTime = System.currentTimeMillis();
                // 3.1 根据构造计算 p[m][j]
                double[][] p = Seru_cacu.computePFromSeruConfig_V0(cfg, Seru_p);

                // 3.2 切换时间 s[m][n][j]
                double[][][] s = Seru_cacu.buildS(cfg.serus.size(), Seru_p.M, Seru_p.batchInfo); // 返回 1-based 且含 dummy 的 s[m][n][j]

                // 3.3 求解该构造下的 MIP
                Result r = null;
                if (s_flag){
                    r = solveOnce(p, s);
                }else{
                    if(B_s_flag){
                        r = solveOnce_No_S(p, Seru_p.conflictPairs);
                    }else{
                        r = solveOnce_No_S(p);
                    }
//                    r = solveOnce_No_S_gap(p, Seru_p.CmaxInit,Seru_p.Gap, Seru_p.StopTime);
                }
                endTime = System.currentTimeMillis();
                if (print_flag)
                    System.out.println("耗时 = " + (endTime-startTime) + "ms, 进度："+idx+"/"+Seru_p.configs.size());
                r.config = cfg;
                r.idx = idx;
                r.problem = Seru_p;
                results[idx] = r;
                // 打印结果
//                String cmaxStr = Double.isFinite(r.Cmax) ? String.format("%.2f", r.Cmax) : "不可行";
//                System.out.println("\n=== 构造 #" + idx + " | Cmax = " + cmaxStr + " ===");
//                System.out.println("SeruConfig = " + cfg);

                // 若可行，打印该构造下的分配
                if (Double.isFinite(r.Cmax)) {
//                    Seru_output.printSolution(r);
                }

                // 3.4 维护全局最优
                if (best == null || r.Cmax < best.Cmax) {
                    best = r;
                    results_best.clear();
                    results_best.add(best);
                }else if(r.Cmax == best.Cmax){
                    results_best.add(r);
                }
                if (r.Cmax > WorstCmax) {
                    WorstCmax = r.Cmax;
                }
                idx++;
            }

            // 4) 输出全局最优
//            Seru_output.printBestSolution(best);
            results[0] = best;
            double[] V = new double[Seru_p.configs.size()+1];
            int t = 0;
            for (Result r : results){
                if (r != null){
                    r.value = (WorstCmax - r.Cmax) / (WorstCmax - best.Cmax);
                    V[t] = r.value;
                    t+=1;
                }
            }
            return results;

        } catch (Exception e) {
            System.err.println("出现异常: " + e.getMessage());
            e.printStackTrace();
        }
        return new Result[0];
    }



    // ====== 单次求解（给定 p[m][j], s[m][n][j]） ======
    private static Result solveOnce(double[][] p, double[][][] s) throws IloException {
        int M0 = p.length;           // 含 dummy 的批次数（= M+1）
        int J  = p[0].length - 1;    // 赛汝数
        int M_real  = M0 - 1;        // 真实批次数

        IloCplex cplex = new IloCplex();
        cplex.setOut(null); // 静默输出

        // 变量
        IloNumVar[][] z = new IloNumVar[M0][J + 1];          // z_{m j}, m=0..M, j=1..J
        for (int m = 0; m <= M_real; m++) for (int j = 1; j <= J; j++) z[m][j] = cplex.boolVar("z_"+m+"_"+j);

        IloNumVar[][][] x = new IloNumVar[M0][M0][J + 1];    // x_{m n j}, m!=n
        for (int m = 0; m <= M_real; m++) {
            for (int n = 0; n <= M_real; n++) {
                if (m == n) continue;
                for (int j = 1; j <= J; j++) x[m][n][j] = cplex.boolVar("x_"+m+"_"+n+"_"+j);
            }
        }

        IloNumVar[][] C = new IloNumVar[M0][J + 1];          // C_{m j}
        for (int m = 0; m <= M_real; m++) for (int j = 1; j <= J; j++) C[m][j] = cplex.numVar(0, Double.MAX_VALUE, "C_"+m+"_"+j);

        IloNumVar[] xi = new IloNumVar[J + 1];               // ξ_j
        for (int j = 1; j <= J; j++) xi[j] = cplex.numVar(0, Double.MAX_VALUE, "xi_"+j);

        IloNumVar Cmax = cplex.numVar(0, Double.MAX_VALUE, "Cmax");

        // 目标
        cplex.addMinimize(Cmax);

        // Big-M（时间推进约束用）
        double B = 1e7;

        // (22) ∑_m z_{m j} p_{m j} + ξ_j ≤ Cmax
        for (int j = 1; j <= J; j++) {
            IloLinearNumExpr lhs = cplex.linearNumExpr();
            for (int m = 1; m <= M_real; m++) lhs.addTerm(p[m][j], z[m][j]);
            lhs.addTerm(1.0, xi[j]);
            cplex.addLe(lhs, Cmax, "c22_j"+j);
        }

        // (23) ∑_j z_{m j} = 1,  m∈M
        for (int m = 1; m <= M_real; m++) {
            IloLinearNumExpr sum = cplex.linearNumExpr();
            for (int j = 1; j <= J; j++) sum.addTerm(1.0, z[m][j]);
            cplex.addEq(sum, 1.0, "c23_m"+m);
        }

        // (24) z_{0 j} = 1
        for (int j = 1; j <= J; j++) cplex.addEq(z[0][j], 1.0, "c24_j"+j);

        // (25) ξ_j = ∑_{m∈M, n∈M0, m≠n} x_{m n j} s_{m n j}
        for (int j = 1; j <= J; j++) {
            IloLinearNumExpr rhs = cplex.linearNumExpr();
            for (int m = 1; m <= M_real; m++)
                for (int n = 0; n <= M_real; n++)
                    if (m != n && x[m][n][j] != null) rhs.addTerm(s[m][n][j], x[m][n][j]);
            cplex.addEq(xi[j], rhs, "c25_j"+j);
        }

        // (26) z_{m j} = ∑_{n∈M0, n≠m} x_{m n j}
        for (int m = 0; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                IloLinearNumExpr sum = cplex.linearNumExpr();
                for (int n = 0; n <= M_real; n++) if (m != n && x[m][n][j] != null) sum.addTerm(1.0, x[m][n][j]);
                cplex.addEq(z[m][j], sum, "c26_m"+m+"_j"+j);
            }
        }

        // (27) z_{m j} = ∑_{n∈M0, n≠m} x_{n m j}
        for (int m = 0; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                IloLinearNumExpr sum = cplex.linearNumExpr();
                for (int n = 0; n <= M_real; n++) if (m != n && x[n][m][j] != null) sum.addTerm(1.0, x[n][m][j]);
                cplex.addEq(z[m][j], sum, "c27_m"+m+"_j"+j);
            }
        }

        // (28) C_{n j} - C_{m j} + B(1 - x_{m n j}) ≥ s_{m n j} + p_{n j},  m∈M0, n∈M
        for (int j = 1; j <= J; j++) {
            for (int m = 0; m <= M_real; m++) {
                for (int n = 1; n <= M_real; n++) {
                    if (m == n) continue;
                    if (x[m][n][j] == null) continue;
                    IloLinearNumExpr lhs = cplex.linearNumExpr();
                    lhs.addTerm(1.0, C[n][j]);
                    lhs.addTerm(-1.0, C[m][j]);
                    lhs.addTerm(-B, x[m][n][j]);
                    lhs.setConstant(B);
                    double rhs = s[m][n][j] + p[n][j];
                    cplex.addGe(lhs, rhs, "c28_m"+m+"_n"+n+"_j"+j);
                }
            }
        }

        // (29) C_{0 j} = 0
        for (int j = 1; j <= J; j++) cplex.addEq(C[0][j], 0.0, "c29_j"+j);

        // 求解
        Result res = new Result();
        res.J = J; res.M = M_real;
        if (cplex.solve()) {
            res.Cmax = cplex.getObjValue();
            res.batchToSeru = new int[M_real + 1]; // 1..M
            for (int m = 1; m <= M_real; m++) {
                int asg = -1;
                for (int j = 1; j <= J; j++) {
                    if (cplex.getValue(z[m][j]) > 0.5) { asg = j; break; }
                }
                res.batchToSeru[m] = asg;
            }
        } else {
            res.Cmax = Double.POSITIVE_INFINITY;
            res.batchToSeru = new int[M_real + 1];
        }
        cplex.end();
        return res;
    }
    // ====== 单次求解（给定 p[m][j], s[m][n][j]） ======
    private static Result solveOnce_No_S(double[][] p) throws IloException {
        int M0 = p.length;           // 含 dummy 的批次数（= M+1）
        int J  = p[0].length - 1;    // 赛汝数
        int M_real  = M0 - 1;        // 真实批次数

        IloCplex cplex = new IloCplex();
        cplex.setOut(null); // 静默输出

        // 变量
        IloNumVar[][] z = new IloNumVar[M0][J + 1];          // z_{m j}, m=0..M, j=1..J
        for (int m = 0; m <= M_real; m++) for (int j = 1; j <= J; j++) z[m][j] = cplex.boolVar("z_"+m+"_"+j);

        IloNumVar[][][] x = new IloNumVar[M0][M0][J + 1];    // x_{m n j}, m!=n
        for (int m = 0; m <= M_real; m++) {
            for (int n = 0; n <= M_real; n++) {
                if (m == n) continue;
                for (int j = 1; j <= J; j++) x[m][n][j] = cplex.boolVar("x_"+m+"_"+n+"_"+j);
            }
        }

        IloNumVar[][] C = new IloNumVar[M0][J + 1];          // C_{m j}
        for (int m = 0; m <= M_real; m++) for (int j = 1; j <= J; j++) C[m][j] = cplex.numVar(0, Double.MAX_VALUE, "C_"+m+"_"+j);

//        IloNumVar[] xi = new IloNumVar[J + 1];               // ξ_j
//        for (int j = 1; j <= J; j++) xi[j] = cplex.numVar(0, Double.MAX_VALUE, "xi_"+j);

        IloNumVar Cmax = cplex.numVar(0, Double.MAX_VALUE, "Cmax");

        // 目标
        cplex.addMinimize(Cmax);

        // Big-M（时间推进约束用）
        double B = 1e7;

        // (22) ∑_m z_{m j} p_{m j} + ξ_j ≤ Cmax
        for (int j = 1; j <= J; j++) {
            IloLinearNumExpr lhs = cplex.linearNumExpr();
            for (int m = 1; m <= M_real; m++) lhs.addTerm(p[m][j], z[m][j]);
//            lhs.addTerm(1.0, xi[j]);
            cplex.addLe(lhs, Cmax, "c22_j"+j);
        }

        // (23) ∑_j z_{m j} = 1,  m∈M
        for (int m = 1; m <= M_real; m++) {
            IloLinearNumExpr sum = cplex.linearNumExpr();
            for (int j = 1; j <= J; j++) sum.addTerm(1.0, z[m][j]);
            cplex.addEq(sum, 1.0, "c23_m"+m);
        }

        // (24) z_{0 j} = 1
        for (int j = 1; j <= J; j++) cplex.addEq(z[0][j], 1.0, "c24_j"+j);

        // (25) ξ_j = ∑_{m∈M, n∈M0, m≠n} x_{m n j} s_{m n j}
//        for (int j = 1; j <= J; j++) {
//            IloLinearNumExpr rhs = cplex.linearNumExpr();
//            for (int m = 1; m <= M_real; m++)
//                for (int n = 0; n <= M_real; n++)
//                    if (m != n && x[m][n][j] != null) rhs.addTerm(s[m][n][j], x[m][n][j]);
//            cplex.addEq(xi[j], rhs, "c25_j"+j);
//        }

        // (26) z_{m j} = ∑_{n∈M0, n≠m} x_{m n j}
        for (int m = 0; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                IloLinearNumExpr sum = cplex.linearNumExpr();
                for (int n = 0; n <= M_real; n++) if (m != n && x[m][n][j] != null) sum.addTerm(1.0, x[m][n][j]);
                cplex.addEq(z[m][j], sum, "c26_m"+m+"_j"+j);
            }
        }

        // (27) z_{m j} = ∑_{n∈M0, n≠m} x_{n m j}
        for (int m = 0; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                IloLinearNumExpr sum = cplex.linearNumExpr();
                for (int n = 0; n <= M_real; n++) if (m != n && x[n][m][j] != null) sum.addTerm(1.0, x[n][m][j]);
                cplex.addEq(z[m][j], sum, "c27_m"+m+"_j"+j);
            }
        }

        // (28) C_{n j} - C_{m j} + B(1 - x_{m n j}) ≥ s_{m n j} + p_{n j},  m∈M0, n∈M
        for (int j = 1; j <= J; j++) {
            for (int m = 0; m <= M_real; m++) {
                for (int n = 1; n <= M_real; n++) {
                    if (m == n) continue;
                    if (x[m][n][j] == null) continue;
                    IloLinearNumExpr lhs = cplex.linearNumExpr();
                    lhs.addTerm(1.0, C[n][j]);
                    lhs.addTerm(-1.0, C[m][j]);
                    lhs.addTerm(-B, x[m][n][j]);
                    lhs.setConstant(B);
                    double rhs = p[n][j];
                    cplex.addGe(lhs, rhs, "c28_m"+m+"_n"+n+"_j"+j);
                }
            }
        }

        // (29) C_{0 j} = 0
        for (int j = 1; j <= J; j++) cplex.addEq(C[0][j], 0.0, "c29_j"+j);

        // 求解
        Result res = new Result();
        res.J = J; res.M = M_real;
        if (cplex.solve()) {
            res.Cmax = cplex.getObjValue();
            res.batchToSeru = new int[M_real + 1]; // 1..M
            for (int m = 1; m <= M_real; m++) {
                int asg = -1;
                for (int j = 1; j <= J; j++) {
                    if (cplex.getValue(z[m][j]) > 0.5) { asg = j; break; }
                }
                res.batchToSeru[m] = asg;
            }
        } else {
            res.Cmax = Double.POSITIVE_INFINITY;
            res.batchToSeru = new int[M_real + 1];
        }
        cplex.end();
        return res;
    }
    // ====== 单次求解（给定 p[m][j], s[m][n][j]） ======
    private static Result solveOnce_No_S(double[][] p, List<int[]> conflictPairs) throws IloException {
        int M0 = p.length;           // 含 dummy 的批次数（= M+1）
        int J  = p[0].length - 1;    // 赛汝数
        int M_real  = M0 - 1;        // 真实批次数

        IloCplex cplex = new IloCplex();
        cplex.setOut(null); // 静默输出

        // 变量
        IloNumVar[][] z = new IloNumVar[M0][J + 1];          // z_{m j}, m=0..M, j=1..J
        for (int m = 0; m <= M_real; m++) for (int j = 1; j <= J; j++) z[m][j] = cplex.boolVar("z_"+m+"_"+j);
        //不兼容批次对约束
        for (int[] pair : conflictPairs) {
            assert pair[0] >= 1 && pair[0] <= M_real;
            assert pair[1] >= 1 && pair[1] <= M_real;
            for (int j = 1; j <= J; j++) {
                IloLinearNumExpr expr = cplex.linearNumExpr();
                expr.addTerm(1.0, z[pair[0]][j]);
                expr.addTerm(1.0, z[pair[1]][j]);
                cplex.addLe(expr, 1.0, "conflict_"+pair[0]+"_"+pair[1]+"_j"+j);
            }
        }

        IloNumVar[][][] x = new IloNumVar[M0][M0][J + 1];    // x_{m n j}, m!=n
        for (int m = 0; m <= M_real; m++) {
            for (int n = 0; n <= M_real; n++) {
                if (m == n) continue;
                for (int j = 1; j <= J; j++) x[m][n][j] = cplex.boolVar("x_"+m+"_"+n+"_"+j);
            }
        }

        IloNumVar[][] C = new IloNumVar[M0][J + 1];          // C_{m j}
        for (int m = 0; m <= M_real; m++) for (int j = 1; j <= J; j++) C[m][j] = cplex.numVar(0, Double.MAX_VALUE, "C_"+m+"_"+j);

//        IloNumVar[] xi = new IloNumVar[J + 1];               // ξ_j
//        for (int j = 1; j <= J; j++) xi[j] = cplex.numVar(0, Double.MAX_VALUE, "xi_"+j);

        IloNumVar Cmax = cplex.numVar(0, Double.MAX_VALUE, "Cmax");

        // 目标
        cplex.addMinimize(Cmax);

        // Big-M（时间推进约束用）
        double B = 1e7;

        // (22) ∑_m z_{m j} p_{m j} + ξ_j ≤ Cmax
        for (int j = 1; j <= J; j++) {
            IloLinearNumExpr lhs = cplex.linearNumExpr();
            for (int m = 1; m <= M_real; m++) lhs.addTerm(p[m][j], z[m][j]);
//            lhs.addTerm(1.0, xi[j]);
            cplex.addLe(lhs, Cmax, "c22_j"+j);
        }

        // (23) ∑_j z_{m j} = 1,  m∈M
        for (int m = 1; m <= M_real; m++) {
            IloLinearNumExpr sum = cplex.linearNumExpr();
            for (int j = 1; j <= J; j++) sum.addTerm(1.0, z[m][j]);
            cplex.addEq(sum, 1.0, "c23_m"+m);
        }

        // (24) z_{0 j} = 1
        for (int j = 1; j <= J; j++) cplex.addEq(z[0][j], 1.0, "c24_j"+j);

        // (25) ξ_j = ∑_{m∈M, n∈M0, m≠n} x_{m n j} s_{m n j}
//        for (int j = 1; j <= J; j++) {
//            IloLinearNumExpr rhs = cplex.linearNumExpr();
//            for (int m = 1; m <= M_real; m++)
//                for (int n = 0; n <= M_real; n++)
//                    if (m != n && x[m][n][j] != null) rhs.addTerm(s[m][n][j], x[m][n][j]);
//            cplex.addEq(xi[j], rhs, "c25_j"+j);
//        }

        // (26) z_{m j} = ∑_{n∈M0, n≠m} x_{m n j}
        for (int m = 0; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                IloLinearNumExpr sum = cplex.linearNumExpr();
                for (int n = 0; n <= M_real; n++) if (m != n && x[m][n][j] != null) sum.addTerm(1.0, x[m][n][j]);
                cplex.addEq(z[m][j], sum, "c26_m"+m+"_j"+j);
            }
        }

        // (27) z_{m j} = ∑_{n∈M0, n≠m} x_{n m j}
        for (int m = 0; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                IloLinearNumExpr sum = cplex.linearNumExpr();
                for (int n = 0; n <= M_real; n++) if (m != n && x[n][m][j] != null) sum.addTerm(1.0, x[n][m][j]);
                cplex.addEq(z[m][j], sum, "c27_m"+m+"_j"+j);
            }
        }

        // (28) C_{n j} - C_{m j} + B(1 - x_{m n j}) ≥ s_{m n j} + p_{n j},  m∈M0, n∈M
        for (int j = 1; j <= J; j++) {
            for (int m = 0; m <= M_real; m++) {
                for (int n = 1; n <= M_real; n++) {
                    if (m == n) continue;
                    if (x[m][n][j] == null) continue;
                    IloLinearNumExpr lhs = cplex.linearNumExpr();
                    lhs.addTerm(1.0, C[n][j]);
                    lhs.addTerm(-1.0, C[m][j]);
                    lhs.addTerm(-B, x[m][n][j]);
                    lhs.setConstant(B);
                    double rhs = p[n][j];
                    cplex.addGe(lhs, rhs, "c28_m"+m+"_n"+n+"_j"+j);
                }
            }
        }

        // (29) C_{0 j} = 0
        for (int j = 1; j <= J; j++) cplex.addEq(C[0][j], 0.0, "c29_j"+j);

        // 求解
        Result res = new Result();
        res.J = J; res.M = M_real;
        if (cplex.solve()) {
            res.Cmax = cplex.getObjValue();
            res.batchToSeru = new int[M_real + 1]; // 1..M
            for (int m = 1; m <= M_real; m++) {
                int asg = -1;
                for (int j = 1; j <= J; j++) {
                    if (cplex.getValue(z[m][j]) > 0.5) { asg = j; break; }
                }
                res.batchToSeru[m] = asg;
            }
        } else {
            res.Cmax = Double.POSITIVE_INFINITY;
            res.batchToSeru = new int[M_real + 1];
        }
        cplex.end();
        return res;
    }
    // Solve with an explicit upper bound and relative MIP gap.
    // public static Result solveOnce_No_S(double[][] p) throws IloException {
    public static Result solveOnce_No_S_gap(double[][] p, double ubCmax, double relativeGap,double StopTime) throws IloException {
        int M0 = p.length;           // 含 dummy 的批次数（= M+1）
        int J  = p[0].length - 1;    // 赛汝数
        int M_real  = M0 - 1;        // 真实批次数

        IloCplex cplex = new IloCplex();
        cplex.setOut(null); // 静默输出

        // Solver limits
        cplex.setParam(Param.Threads, 4);  // 并行核心数

        cplex.setParam(Param.TimeLimit, StopTime);              // 原有
        cplex.setParam(Param.MIP.Tolerances.MIPGap, relativeGap); // 原有

        // Optional node limit.
//        cplex.setParam(IloCplex.Param.MIP.Limits.Nodes, 200000);

        // Optional deterministic parallel mode.
//        cplex.setParam(IloCplex.Param.Parallel, IloCplex.ParallelMode.Deterministic);

        // Precompute max p_{mj} for Big-M.
        double maxP = 0.0;
        for (int m = 1; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                if (p[m][j] > maxP) {
                    maxP = p[m][j];
                }
            }
        }
        // Derive Big-M from the incumbent bound and maximum processing time.
        double B = ubCmax + maxP;


        // 变量
        IloNumVar[][] z = new IloNumVar[M0][J + 1];          // z_{m j}, m=0..M, j=1..J
        for (int m = 0; m <= M_real; m++) for (int j = 1; j <= J; j++) z[m][j] = cplex.boolVar("z_"+m+"_"+j);

        IloNumVar[][][] x = new IloNumVar[M0][M0][J + 1];    // x_{m n j}, m!=n
        for (int m = 0; m <= M_real; m++) {
            for (int n = 0; n <= M_real; n++) {
                if (m == n) continue;
                for (int j = 1; j <= J; j++) x[m][n][j] = cplex.boolVar("x_"+m+"_"+n+"_"+j);
            }
        }

        IloNumVar[][] C = new IloNumVar[M0][J + 1];          // C_{m j}
        for (int m = 0; m <= M_real; m++) for (int j = 1; j <= J; j++) C[m][j] = cplex.numVar(0, ubCmax, "C_"+m+"_"+j);

//        IloNumVar[] xi = new IloNumVar[J + 1];               // ξ_j
//        for (int j = 1; j <= J; j++) xi[j] = cplex.numVar(0, Double.MAX_VALUE, "xi_"+j);

        IloNumVar Cmax = cplex.numVar(0, ubCmax, "Cmax");

        // 2. 添加上界 (UB) 约束
        //    (使用 Cmax <= UB 的方法)
        cplex.addLe(Cmax, ubCmax, "c_UpperBound");

        // 目标
        cplex.addMinimize(Cmax);

        // Big-M（时间推进约束用）
//        double B = 1e7;

        // (22) ∑_m z_{m j} p_{m j} + ξ_j ≤ Cmax
        for (int j = 1; j <= J; j++) {
            IloLinearNumExpr lhs = cplex.linearNumExpr();
            for (int m = 1; m <= M_real; m++) lhs.addTerm(p[m][j], z[m][j]);
//            lhs.addTerm(1.0, xi[j]);
            cplex.addLe(lhs, Cmax, "c22_j"+j);
        }

        // (23) ∑_j z_{m j} = 1,  m∈M
        for (int m = 1; m <= M_real; m++) {
            IloLinearNumExpr sum = cplex.linearNumExpr();
            for (int j = 1; j <= J; j++) sum.addTerm(1.0, z[m][j]);
            cplex.addEq(sum, 1.0, "c23_m"+m);
        }

        // (24) z_{0 j} = 1
        for (int j = 1; j <= J; j++) cplex.addEq(z[0][j], 1.0, "c24_j"+j);

        // (25) ξ_j = ∑_{m∈M, n∈M0, m≠n} x_{m n j} s_{m n j}
//        for (int j = 1; j <= J; j++) {
//            IloLinearNumExpr rhs = cplex.linearNumExpr();
//            for (int m = 1; m <= M_real; m++)
//                for (int n = 0; n <= M_real; n++)
//                    if (m != n && x[m][n][j] != null) rhs.addTerm(s[m][n][j], x[m][n][j]);
//            cplex.addEq(xi[j], rhs, "c25_j"+j);
//        }

        // (26) z_{m j} = ∑_{n∈M0, n≠m} x_{m n j}
        for (int m = 0; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                IloLinearNumExpr sum = cplex.linearNumExpr();
                for (int n = 0; n <= M_real; n++) if (m != n && x[m][n][j] != null) sum.addTerm(1.0, x[m][n][j]);
                cplex.addEq(z[m][j], sum, "c26_m"+m+"_j"+j);
            }
        }

        // (27) z_{m j} = ∑_{n∈M0, n≠m} x_{n m j}
        for (int m = 0; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                IloLinearNumExpr sum = cplex.linearNumExpr();
                for (int n = 0; n <= M_real; n++) if (m != n && x[n][m][j] != null) sum.addTerm(1.0, x[n][m][j]);
                cplex.addEq(z[m][j], sum, "c27_m"+m+"_j"+j);
            }
        }

        // (28) C_{n j} - C_{m j} + B(1 - x_{m n j}) ≥ s_{m n j} + p_{n j},  m∈M0, n∈M
        for (int j = 1; j <= J; j++) {
            for (int m = 0; m <= M_real; m++) {
                for (int n = 1; n <= M_real; n++) {
                    if (m == n) continue;
                    if (x[m][n][j] == null) continue;
                    IloLinearNumExpr lhs = cplex.linearNumExpr();
                    lhs.addTerm(1.0, C[n][j]);
                    lhs.addTerm(-1.0, C[m][j]);
                    lhs.addTerm(-B, x[m][n][j]);
                    lhs.setConstant(B);
                    double rhs = p[n][j];
                    cplex.addGe(lhs, rhs, "c28_m"+m+"_n"+n+"_j"+j);
                }
            }
        }

        // (29) C_{0 j} = 0
        for (int j = 1; j <= J; j++) cplex.addEq(C[0][j], 0.0, "c29_j"+j);

        // 求解
        Result res = new Result();
        res.J = J; res.M = M_real;
        if (cplex.solve()) {
            res.Cmax = cplex.getObjValue();
            res.batchToSeru = new int[M_real + 1]; // 1..M
            for (int m = 1; m <= M_real; m++) {
                int asg = -1;
                for (int j = 1; j <= J; j++) {
                    if (cplex.getValue(z[m][j]) > 0.5) { asg = j; break; }
                }
                res.batchToSeru[m] = asg;
            }
        } else {
//            res.Cmax = Double.POSITIVE_INFINITY;
            res.Cmax = ubCmax;
            res.batchToSeru = new int[M_real + 1];
        }
        cplex.end();
        return res;
    }

    private static Result solveOnce_No_S_Simplified(double[][] p) throws IloException {
        int M0 = p.length;
        int J = p[0].length - 1;
        int M_real = M0 - 1;

        IloCplex cplex = new IloCplex();
        cplex.setOut(null);

        // 1. 变量 (只需要 z 和 Cmax)
        IloNumVar[][] z = new IloNumVar[M_real + 1][J + 1]; // z_{m j}, m=1..M, j=1..J
        for (int m = 1; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                z[m][j] = cplex.boolVar("z_" + m + "_" + j);
            }
        }

        IloNumVar Cmax = cplex.numVar(0, Double.MAX_VALUE, "Cmax");

        // 2. 目标
        cplex.addMinimize(Cmax);

        // 3. 约束 (只需要 22 和 23)

        // (22) ∑_m z_{m j} p_{m j} ≤ Cmax
        for (int j = 1; j <= J; j++) {
            IloLinearNumExpr lhs = cplex.linearNumExpr();
            for (int m = 1; m <= M_real; m++) {
                lhs.addTerm(p[m][j], z[m][j]);
            }
            cplex.addLe(lhs, Cmax, "c22_j" + j);
        }

        // (23) ∑_j z_{m j} = 1,  m∈M
        for (int m = 1; m <= M_real; m++) {
            IloLinearNumExpr sum = cplex.linearNumExpr();
            for (int j = 1; j <= J; j++) {
                sum.addTerm(1.0, z[m][j]);
            }
            cplex.addEq(sum, 1.0, "c23_m" + m);
        }

        // 4. 求解
        Result res = new Result();
        res.J = J; res.M = M_real;
        if (cplex.solve()) {
            res.Cmax = cplex.getObjValue();
            res.batchToSeru = new int[M_real + 1]; // 1..M
            for (int m = 1; m <= M_real; m++) {
                int asg = -1;
                for (int j = 1; j <= J; j++) {
                    if (cplex.getValue(z[m][j]) > 0.5) {
                        asg = j;
                        break;
                    }
                }
                res.batchToSeru[m] = asg;
            }
        } else {
            res.Cmax = Double.POSITIVE_INFINITY;
            res.batchToSeru = new int[M_real + 1];
        }
        cplex.end();
        return res;
    }
    // private static Result solveOnce_No_S_Simplified(double[][] p) throws IloException {
    public static Result solveOnce_No_S_Simplified_Gap(double[][] p, double ubCmax, double relativeGap) throws IloException {
        int M0 = p.length;
        int J = p[0].length - 1;
        int M_real = M0 - 1;

        IloCplex cplex = new IloCplex();
        cplex.setOut(null); // 静默输出

        // Apply the relative MIP gap and incumbent upper bound.

        // 1. 设置相对MIP Gap
        cplex.setParam(Param.MIP.Tolerances.MIPGap, relativeGap);


        // 1. 变量 (只需要 z 和 Cmax)
        IloNumVar[][] z = new IloNumVar[M_real + 1][J + 1]; // z_{m j}, m=1..M, j=1..J
        for (int m = 1; m <= M_real; m++) {
            for (int j = 1; j <= J; j++) {
                z[m][j] = cplex.boolVar("z_" + m + "_" + j);
            }
        }

        IloNumVar Cmax = cplex.numVar(0, Double.MAX_VALUE, "Cmax");

        // 2. 添加上界 (UB) 约束
        cplex.addLe(Cmax, ubCmax, "c_UpperBound");


        // 2. 目标
        cplex.addMinimize(Cmax);

        // 3. 约束 (只需要 22 和 23)

        // (22) ∑_m z_{m j} p_{m j} ≤ Cmax
        for (int j = 1; j <= J; j++) {
            IloLinearNumExpr lhs = cplex.linearNumExpr();
            for (int m = 1; m <= M_real; m++) {
                lhs.addTerm(p[m][j], z[m][j]);
            }
            cplex.addLe(lhs, Cmax, "c22_j" + j);
        }

        // (23) ∑_j z_{m j} = 1,  m∈M
        for (int m = 1; m <= M_real; m++) {
            IloLinearNumExpr sum = cplex.linearNumExpr();
            for (int j = 1; j <= J; j++) {
                sum.addTerm(1.0, z[m][j]);
            }
            cplex.addEq(sum, 1.0, "c23_m" + m);
        }

        // 4. 求解
        Result res = new Result();
        res.J = J;
        res.M = M_real;
        if (cplex.solve()) {
            res.Cmax = cplex.getObjValue();
            res.batchToSeru = new int[M_real + 1]; // 1..M
            for (int m = 1; m <= M_real; m++) {
                int asg = -1;
                for (int j = 1; j <= J; j++) {
                    // 检查 z[m][j] 是否为 null (如果 M_real < 1 可能会发生)
                    if (z[m][j] != null && cplex.getValue(z[m][j]) > 0.5) {
                        asg = j;
                        break;
                    }
                }
                res.batchToSeru[m] = asg;
            }
        } else {
            res.Cmax = Double.POSITIVE_INFINITY;
            res.batchToSeru = new int[M_real + 1];
        }
        cplex.end();
        return res;
    }
}
