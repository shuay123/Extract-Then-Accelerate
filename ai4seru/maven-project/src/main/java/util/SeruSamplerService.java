package util;

import model.Seru;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.IOException;
import java.io.InputStream;
import java.io.FileInputStream;
import java.util.*;
import java.util.stream.Collectors;

public class SeruSamplerService {

    // ======== 主要入口 ========
    public static Result sample(Config cfg) throws Exception {
        filecontent F = new filecontent();
        F.GetContent(cfg);

        return GetSamples(F.allBatches, F.sd, F.workerCoeffs,cfg);



//        Objects.requireNonNull(cfg.excelPath, "excelPath 不能为空");
//        Objects.requireNonNull(cfg.sheetNameBatch, "sheetNameBatch 不能为空");
//        Objects.requireNonNull(cfg.sheetNameSkill, "sheetNameSkill 不能为空");
//
//        if (cfg.W < 1) throw new IllegalArgumentException("W 必须 >= 1");
//        if (cfg.J < 1) throw new IllegalArgumentException("J 必须 >= 1");
//        if (cfg.workerMin > cfg.workerMax) throw new IllegalArgumentException("workerMin <= workerMax");
//        if (cfg.batchMin > cfg.batchMax) throw new IllegalArgumentException("batchMin <= batchMax");
//        if (cfg.W > (cfg.workerMax - cfg.workerMin + 1))
//            throw new IllegalArgumentException("W 超过工人范围大小");
//        if (cfg.J > (cfg.batchMax - cfg.batchMin + 1))
//            throw new IllegalArgumentException("J 超过批次范围大小");
//
//        Random rnd = (cfg.seed == null) ? new Random() : new Random(cfg.seed);
//
//        try (InputStream in = new FileInputStream(cfg.excelPath);
//             XSSFWorkbook wb = new XSSFWorkbook(in)) {
//
//            XSSFSheet batchSheet = wb.getSheet(cfg.sheetNameBatch);
//            XSSFSheet skillSheet = wb.getSheet(cfg.sheetNameSkill);
//            if (batchSheet == null) throw new IllegalStateException("未找到 sheet: " + cfg.sheetNameBatch);
//            if (skillSheet == null) throw new IllegalStateException("未找到 sheet: " + cfg.sheetNameSkill);
//
//            // 1) 批次表：按前三列读取（int,int,int），忽略表头行(默认第0行是表头)
//            List<BatchInfo> allBatches = readBatchesFixed3Cols(batchSheet);
//
//            // 2) 技能表：首行=产品类型(int，从第2列起)，首列=工人ID
//            SkillData sd = readWorkerSkillsByHeaderInts(skillSheet);
//
//            // 3) 抽取工人
//            int[] workerIds = sampleRangeDistinct(cfg.workerMin, cfg.workerMax, cfg.W, rnd);
//
//            // 4) 抽取批次（需存在于表内，且编号落在区间内）
//            List<Integer> pool = new ArrayList<>();
//            for (BatchInfo b : allBatches) {
//                if (b.batchId >= cfg.batchMin && b.batchId <= cfg.batchMax) {
//                    pool.add(b.batchId);
//                }
//            }
//            pool = distinctSorted(pool);
//            if (pool.size() < cfg.J) {
//                throw new IllegalStateException("可用批次数不足，需 J=" + cfg.J + "，实际仅有 " + pool.size());
//            }
//            List<Integer> picked = pickFirstN(shuffleCopy(pool, rnd), cfg.J);
//            Collections.sort(picked);
//
//            // 5) 对应批次信息（保持与 picked 顺序一致）
//            Map<Integer, BatchInfo> byId = new HashMap<>();
//            for (BatchInfo b : allBatches) byId.put(b.batchId, b);
//            List<BatchInfo> chosen = new ArrayList<>();
//            for (int bid : picked) chosen.add(byId.get(bid));
//
//            // 6) 组装 [W, J] 熟练度矩阵（按批次的产品类型列取值）
//            double[][] prof = new double[workerIds.length][chosen.size()];
//            for (int i = 0; i < workerIds.length; i++) {
//                int wid = workerIds[i];
//                Map<Integer, Double> skillMap = sd.worker2ptype2skill.getOrDefault(wid, Collections.emptyMap());
//                for (int j = 0; j < chosen.size(); j++) {
//                    int ptype = chosen.get(j).productType; // int
//                    prof[i][j] = skillMap.getOrDefault(ptype, 0.0);
//                }
//            }
//
//            return new Result(chosen, workerIds, prof);
//        }
    }

    // ======== 读取：批次固定三列 ========
    public static class filecontent{
        public List<BatchInfo> allBatches;  // 批次表
        public SkillData sd;  // 技能表
        public List<WorkerCoeffInfo> workerCoeffs;  // 多能工系数表

        public void GetContent(Config cfg) throws Exception {
            Objects.requireNonNull(cfg.excelPath, "excelPath 不能为空");
            Objects.requireNonNull(cfg.sheetNameBatch, "sheetNameBatch 不能为空");
            Objects.requireNonNull(cfg.sheetNameSkill, "sheetNameSkill 不能为空");

            if (cfg.W < 1) throw new IllegalArgumentException("W 必须 >= 1");
            if (cfg.J < 1) throw new IllegalArgumentException("J 必须 >= 1");
            if (cfg.workerMin > cfg.workerMax) throw new IllegalArgumentException("workerMin <= workerMax");
            if (cfg.batchMin > cfg.batchMax) throw new IllegalArgumentException("batchMin <= batchMax");
            if (cfg.W > (cfg.workerMax - cfg.workerMin + 1))
                throw new IllegalArgumentException("W 超过工人范围大小");
            if (cfg.J > (cfg.batchMax - cfg.batchMin + 1))
                throw new IllegalArgumentException("J 超过批次范围大小");

            try (InputStream in = new FileInputStream(cfg.excelPath);
                 XSSFWorkbook wb = new XSSFWorkbook(in)) {

                XSSFSheet batchSheet = wb.getSheet(cfg.sheetNameBatch);
                XSSFSheet skillSheet = wb.getSheet(cfg.sheetNameSkill);
                if (batchSheet == null) throw new IllegalStateException("未找到 sheet: " + cfg.sheetNameBatch);
                if (skillSheet == null) throw new IllegalStateException("未找到 sheet: " + cfg.sheetNameSkill);

                // 1) 批次表：按前三列读取（int,int,int），忽略表头行(默认第0行是表头)
                this.allBatches = readBatchesFixed3Cols(batchSheet);

                // 2) 技能表：首行=产品类型(int，从第2列起)，首列=工人ID
                this.sd = readWorkerSkillsByHeaderInts(skillSheet);

                // 3) 多能工系数表：按前两列读取（int,double），忽略表头行(默认第0行是表头)
                this.workerCoeffs = readWorkerCoeffsFixed2Cols(wb.getSheet(cfg.sheetNameCoeff));

            }
        }

    }

    public static Result GetSamples(List<BatchInfo> allBatches, SkillData sd, List<WorkerCoeffInfo> allWorkerCoeff,Config cfg){
        Random rnd = (cfg.seed == null) ? new Random() : new Random(cfg.seed);
        // 3) 抽取工人
        int[] workerIds = sampleRangeDistinct(cfg.workerMin, cfg.workerMax, cfg.W, rnd);

        // 4) 抽取批次（需存在于表内，且编号落在区间内）
        List<Integer> pool = new ArrayList<>();
        for (BatchInfo b : allBatches) {
            if (b.batchId >= cfg.batchMin && b.batchId <= cfg.batchMax) {
                pool.add(b.batchId);
            }
        }
        pool = distinctSorted(pool);
        if (pool.size() < cfg.J) {
            throw new IllegalStateException("可用批次数不足，需 J=" + cfg.J + "，实际仅有 " + pool.size());
        }
        List<Integer> picked = pickFirstN(shuffleCopy(pool, rnd), cfg.J);
        Collections.sort(picked);


        // 5) 对应批次信息（保持与 picked 顺序一致）
        Map<Integer, BatchInfo> byId = new HashMap<>();
        for (BatchInfo b : allBatches) byId.put(b.batchId, b);
        List<BatchInfo> chosen = new ArrayList<>();
        for (int bid : picked) chosen.add(byId.get(bid));

        // 6) 组装 [W, J] 熟练度矩阵（按批次的产品类型列取值）
        double[][] prof = new double[workerIds.length][chosen.size()];
        double[][] profToType = new double[workerIds.length][5];
        for (int i = 0; i < workerIds.length; i++) {
            int wid = workerIds[i];
            Map<Integer, Double> skillMap = sd.worker2ptype2skill.getOrDefault(wid, Collections.emptyMap());
            for (int j = 0; j < chosen.size(); j++) {
                int ptype = chosen.get(j).productType; // int
                prof[i][j] = skillMap.getOrDefault(ptype, 0.0);

            }
            for (int j = 0; j < 5; j++) {
                profToType[i][j] = skillMap.getOrDefault(j+1, 0.0);
            }
        }
        int[] batchIds = new int[chosen.size()];
        int[] batchSize = new int[chosen.size()+1];
        for (int i = 0; i < chosen.size(); i++) {
            batchIds[i] = chosen.get(i).batchId;
            batchSize[i+1] = chosen.get(i).batchSize;
        }
        batchSize[0] = 0;
        double[] workerCoefficients = new double[workerIds.length+1];
        for (int i = 0; i < workerIds.length; i++) {
            workerCoefficients[i+1] = allWorkerCoeff.get(workerIds[i]-1).coefficient;
        }

        return new Result(chosen, workerIds, batchIds, batchSize, prof, workerCoefficients, cfg.Gap, profToType);
    }

    public static Result GetSamplesFromFile(String filePath) throws IOException {

        try (FileInputStream fis = new FileInputStream(filePath);
             Workbook workbook = new XSSFWorkbook(fis)) {

            // ── 1. 读取 workerInfo sheet ──────────────────────────────────────
            // 列：workerID | 1 | 2 | 3 | 4 | 5 | Coefficients
            Sheet workerSheet = workbook.getSheet("workerInfo");
            List<Integer> workerIdList = new ArrayList<>();
            List<double[]> profToTypeList = new ArrayList<>();
            List<Double> coeffList = new ArrayList<>();

            for (int i = 1; i <= workerSheet.getLastRowNum(); i++) {   // 跳过表头(第0行)
                Row row = workerSheet.getRow(i);
                if (row == null) continue;

                int wid = (int) getNumeric(row, 0);
                workerIdList.add(wid);

                double[] prof5 = new double[5];
                for (int j = 0; j < 5; j++) {
                    prof5[j] = getNumeric(row, j + 1);
                }
                profToTypeList.add(prof5);

                coeffList.add(getNumeric(row, 6));
            }

            int W = workerIdList.size();
            int[] workerIds = workerIdList.stream().mapToInt(Integer::intValue).toArray();

            // profToType[i][j]: 第 i 个工人对第 j+1 类产品的熟练度
            double[][] profToType = new double[W][5];
            for (int i = 0; i < W; i++) profToType[i] = profToTypeList.get(i);

            // workerCoefficients 下标从 1 开始（与 GetSamples 保持一致）
            double[] workerCoefficients = new double[W + 1];
            for (int i = 0; i < W; i++) workerCoefficients[i + 1] = coeffList.get(i);

            // ── 2. 读取 batchInfo sheet ───────────────────────────────────────
            // 列：BatchID | Type | size
            Sheet batchSheet = workbook.getSheet("batchInfo");
            List<BatchInfo> chosen = new ArrayList<>();
            List<Integer> batchIdList = new ArrayList<>();
            List<Integer> batchSizeList = new ArrayList<>();

            for (int i = 1; i <= batchSheet.getLastRowNum(); i++) {
                Row row = batchSheet.getRow(i);
                if (row == null) continue;

                int bid  = (int) getNumeric(row, 0);
                int type = (int) getNumeric(row, 1);
                int size = (int) getNumeric(row, 2);

                BatchInfo b = new BatchInfo(bid, type, size);
                chosen.add(b);

                batchIdList.add(bid);
                batchSizeList.add(size);
            }

            int J = chosen.size();
            int[] batchIds = batchIdList.stream().mapToInt(Integer::intValue).toArray();

            // batchSize 下标从 1 开始，下标 0 固定为 0（与 GetSamples 保持一致）
            int[] batchSize = new int[J + 1];
            batchSize[0] = 0;
            for (int i = 0; i < J; i++) batchSize[i + 1] = batchSizeList.get(i);

            // ── 3. 重建 prof 矩阵 [W × J] ─────────────────────────────────────
            // GetSamples 中该矩阵的赋值被注释掉了（全为 0）；
            // 此处按批次的产品类型从 profToType 中取值来正确重建。
            double[][] prof = new double[W][J];
            for (int i = 0; i < W; i++) {
                for (int j = 0; j < J; j++) {
                    int ptype = chosen.get(j).productType;   // 1~5
                    prof[i][j] = profToType[i][ptype - 1];
                }
            }

            // ── 4. 读取 problem sheet（可选，用于校验或填充 Gap 等字段）────────
            Sheet problemSheet = workbook.getSheet("problem");
            Row problemRow = problemSheet.getRow(1);
            double cmax = getNumeric(problemRow, 4);
            double solvingtime = getNumeric(problemRow, 3);
            // double gap = getNumeric(problemRow, ?); // Gap 未导出到 Excel，需另行处理

            // ── 5. 组装并返回 ─────────────────────────────────────────────────
            return new SeruSamplerService.Result(
                    chosen,
                    workerIds,
                    batchIds,
                    batchSize,
                    prof,
                    workerCoefficients,
                    0.0,        // Gap：原文件未导出，置默认值，按需替换
                    profToType,
                    cmax,
                    solvingtime
            );
        }
    }

    private static double getNumeric(Row row, int col) {
        Cell cell = row.getCell(col);
        if (cell == null) return 0.0;
        return switch (cell.getCellType()) {
            case NUMERIC -> cell.getNumericCellValue();
            case STRING  -> Double.parseDouble(cell.getStringCellValue().trim());
            default      -> 0.0;
        };
    }

    // ======== 读取：批次固定三列 ========
    private static List<BatchInfo> readBatchesFixed3Cols(Sheet s) {
        List<BatchInfo> out = new ArrayList<>();
        int first = s.getFirstRowNum();
        int last  = s.getLastRowNum();
        for (int r = first + 1; r <= last; r++) {            // 跳过第0行表头
            Row row = s.getRow(r);
            if (row == null) continue;
            Integer bid = parseInt(row.getCell(0));
            Integer ptype = parseInt(row.getCell(1));        // 产品类型：整型
            Integer bsize = parseInt(row.getCell(2));
            if (bid == null) continue;
            if (ptype == null) ptype = 0;
            if (bsize == null) bsize = 0;
            out.add(new BatchInfo(bid, ptype, bsize));
        }
        return out;
    }

    // ======== 读取：多能工系数表（前两列int,double） ========
    private static List<WorkerCoeffInfo> readWorkerCoeffsFixed2Cols(Sheet s) {
        List<WorkerCoeffInfo> out = new ArrayList<>();
        int first = s.getFirstRowNum();
        int last  = s.getLastRowNum();
        for (int r = first + 1; r <= last; r++) {            // 跳过第0行表头
            Row row = s.getRow(r);
            if (row == null) continue;
            Integer WorkerId = parseInt(row.getCell(0));
            Double Coeff = parseDouble(row.getCell(1));        // 多能工系数：双精度

            if (WorkerId == null) continue;
            if (Coeff == null) Coeff = 0.0;
            out.add(new WorkerCoeffInfo(WorkerId, Coeff));
        }
        return out;
    }

    // ======== 读取：技能表（首行产品类型=整型，首列工人ID） ========
    private static SkillData readWorkerSkillsByHeaderInts(Sheet s) {
        SkillData sd = new SkillData();
        int first = s.getFirstRowNum();
        Row header = s.getRow(first);
        if (header == null) return sd;

        // 收集“产品类型(整型)所在的列索引”
        List<Integer> ptypeCols = new ArrayList<>();
        List<Integer> ptypeIds  = new ArrayList<>();
        short lastCell = header.getLastCellNum();
        for (int c = 1; c < lastCell; c++) {
            Integer ptype = parseInt(header.getCell(c));
            if (ptype != null) {
                ptypeCols.add(c);
                ptypeIds.add(ptype);
            }
        }

        // 读取每一行：工人ID + 各产品熟练度
        int last = s.getLastRowNum();
        for (int r = first + 1; r <= last; r++) {
            Row row = s.getRow(r);
            if (row == null) continue;
            Integer wid = parseInt(row.getCell(0));
            if (wid == null) continue;

            Map<Integer, Double> map = new HashMap<>();
            for (int k = 0; k < ptypeCols.size(); k++) {
                int c = ptypeCols.get(k);
                int ptype = ptypeIds.get(k);
                Double val = parseDouble(row.getCell(c));
                map.put(ptype, (val == null ? 0.0 : val));
            }
            sd.worker2ptype2skill.put(wid, map);
        }
        return sd;
    }

    // ======== 小工具 ========
    private static Integer parseInt(Cell cell) {
        if (cell == null) return null;
        try {
            return switch (cell.getCellType()) {
                case NUMERIC -> (int) Math.round(cell.getNumericCellValue());
                case STRING  -> {
                    String s = cell.getStringCellValue();
                    if (s == null || s.isBlank()) yield null;
                    int dot = s.indexOf('.');
                    if (dot >= 0) s = s.substring(0, dot);
                    yield Integer.parseInt(s.trim());
                }
                case FORMULA -> {
                    try { yield (int) Math.round(cell.getNumericCellValue()); }
                    catch (Exception e) {
                        String s = cell.getStringCellValue();
                        if (s == null || s.isBlank()) yield null;
                        int dot2 = s.indexOf('.');
                        if (dot2 >= 0) s = s.substring(0, dot2);
                        yield Integer.parseInt(s.trim());
                    }
                }
                default -> null;
            };
        } catch (Exception e) { return null; }
    }

    private static Double parseDouble(Cell cell) {
        if (cell == null) return null;
        try {
            return switch (cell.getCellType()) {
                case NUMERIC -> cell.getNumericCellValue();
                case STRING  -> {
                    String s = cell.getStringCellValue();
                    if (s == null || s.isBlank()) yield null;
                    yield Double.parseDouble(s.trim());
                }
                case FORMULA -> {
                    try { yield cell.getNumericCellValue(); }
                    catch (Exception e) {
                        String s = cell.getStringCellValue();
                        if (s == null || s.isBlank()) yield null;
                        yield Double.parseDouble(s.trim());
                    }
                }
                default -> null;
            };
        } catch (Exception e) { return null; }
    }

    private static int[] sampleRangeDistinct(int min, int max, int n, Random rnd) {
        List<Integer> pool = new ArrayList<>();
        for (int i = min; i <= max; i++) pool.add(i);
        Collections.shuffle(pool, rnd);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = pool.get(i);
        Arrays.sort(out);
        return out;
    }

    private static List<Integer> distinctSorted(List<Integer> a) {
        TreeSet<Integer> s = new TreeSet<>(a);
        return new ArrayList<>(s);
    }

    private static <T> List<T> shuffleCopy(List<T> a, Random rnd) {
        ArrayList<T> b = new ArrayList<>(a);
        Collections.shuffle(b, rnd);
        return b;
    }

    private static <T> List<T> pickFirstN(List<T> a, int n) {
        return new ArrayList<>(a.subList(0, n));
    }

    // ======== 数据结构 ========
    public static class BatchInfo {
        public final int batchId;
        public final int productType; // 已改为整型
        public final int batchSize;
        public BatchInfo(int batchId, int productType, int batchSize) {
            this.batchId = batchId; this.productType = productType; this.batchSize = batchSize;
        }
    }

    // ======== 数据结构 ========
    public static class WorkerCoeffInfo {
        public final int workerId;
        public final double coefficient;
        public WorkerCoeffInfo(int workerId, double coefficient) {
            this.workerId = workerId; this.coefficient = coefficient;
        }
    }

    private static class SkillData {
        // workerId -> (productType -> skill)
        final Map<Integer, Map<Integer, Double>> worker2ptype2skill = new HashMap<>();
    }

    public static class Result {
        public final List<BatchInfo> batches;   // 长度 J
        public final int[] workerIds;           // 长度 W
        public double cmax_init;
        public double cmax;
        public double SolvingTime;
        public double Gap;
        public final int[] batchIds;
        public final int[] batchSize;
        public final double[][] workerProficiencies; // [W, J]
        public final double[][] workerProficienciesToType; // [W, J]
        public final double[] workerCoefficients; // [W]

        public Result(List<BatchInfo> batches, int[] workerIds, int[] batchIds,int[] batchSize, double[][] workerProficiencies, double[] workerCoefficients, double Gap, double[][] workerProficienciesToType) {
            this.batches = batches;
            this.workerIds = workerIds;
            this.batchIds = batchIds;
            this.batchSize = batchSize;
            this.workerProficiencies = workerProficiencies;
            this.workerProficienciesToType = workerProficienciesToType;
            this.workerCoefficients = workerCoefficients;

            this.cmax_init = -500000;
            this.Gap = Gap;
        }
        public Result(List<BatchInfo> batches, int[] workerIds, int[] batchIds,int[] batchSize, double[][] workerProficiencies, double[] workerCoefficients, double Gap, double[][] workerProficienciesToType, double cmax, double solvingTime) {
            this.batches = batches;
            this.workerIds = workerIds;
            this.batchIds = batchIds;
            this.batchSize = batchSize;
            this.workerProficiencies = workerProficiencies;
            this.workerProficienciesToType = workerProficienciesToType;
            this.workerCoefficients = workerCoefficients;

            this.cmax_init = -500000;
            this.Gap = Gap;

            this.cmax = cmax;
            this.SolvingTime = solvingTime;

        }

        /** [J,3]：每行 = (batch_id, product_type(int), batch_size) */
        public int[][] batchTable() {
            int[][] t = new int[batches.size()][3];
            for (int i = 0; i < batches.size(); i++) {
                BatchInfo b = batches.get(i);
                t[i][0] = b.batchId;
                t[i][1] = b.productType;
                t[i][2] = b.batchSize;
            }
            return t;
        }

        /** [W, J+1]：首列 worker_id，其余列为熟练度 */
        public Object[][] workerTable() {
            int W = workerIds.length, J = batches.size();
            Object[][] t = new Object[W][J + 1];
            for (int i = 0; i < W; i++) {
                t[i][0] = workerIds[i];
                for (int j = 0; j < J; j++) {
                    t[i][j + 1] = workerProficiencies[i][j];
                }
            }
            return t;
        }
    }

    // ======== 配置（简洁可直接 new 后改字段） ========
    public static class Config {
        public String excelPath;
        public String sheetNameBatch = "批次与产品类型关系";
        public String sheetNameSkill = "工人与产品类型熟练程度_京东";
        public String sheetNameCoeff = "多能工系数";
        public String ServerURL;
        public String filepath;
        public String outpath;

        public int workerMin = 1, workerMax = 40;
        public int batchMin  = 1, batchMax  = 30;
        public int W = 1, J = 1;
        public Long seed = null;
        public int max_num_of_multiple_task = 10;
        public double Gap = 0.01;
        public double StopTime = 300;
        public double AlphaOfScore = 0.5;
        public double AlphaOfScoreBatch = 0.2;
    }
}

