package util;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.apache.poi.ss.usermodel.Cell;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.temporal.TemporalAccessor;
import java.util.*;

/** 多 Sheet 导出工具：每个 Sheet 一张表（表头 + 多行数据） */
public class ExcelMultiSheetWriter {

    /** 表数据结构：headers 对应表头，rows 每行一个 List<Object> */
    public static class DataTable {
        public final List<String> headers;
        public final List<List<Object>> rows;

        public DataTable(List<String> headers, List<List<Object>> rows) {
            this.headers = headers == null ? Collections.emptyList() : headers;
            this.rows = rows == null ? Collections.emptyList() : rows;
        }
        public static DataTable of(List<String> headers, List<List<Object>> rows) {
            return new DataTable(headers, rows);
        }
    }

    /**
     * 将多份数据写入到同一 Excel 的不同 Sheet。
     * @param sheets  key 为 sheet 名，value 为该表数据
     * @param outPathStr 输出文件路径（.xlsx）
     */
    public static void writeXlsx(Map<String, DataTable> sheets, String outPathStr) throws IOException {
        try (Workbook wb = new XSSFWorkbook()) {
            // 基础样式：表头加粗、日期格式
            CellStyle headerStyle = wb.createCellStyle();
            Font bold = wb.createFont();
            bold.setBold(true);
            headerStyle.setFont(bold);
            headerStyle.setWrapText(false);
            headerStyle.setBorderBottom(BorderStyle.THIN);

            CellStyle dateStyle = wb.createCellStyle();
            CreationHelper helper = wb.getCreationHelper();
            short fmt = helper.createDataFormat().getFormat("yyyy-mm-dd hh:mm:ss");
            dateStyle.setDataFormat(fmt);

            for (Map.Entry<String, DataTable> e : sheets.entrySet()) {
                String sheetName = sanitizeSheetName(e.getKey());
                Sheet sheet = wb.createSheet(sheetName);
                DataTable table = e.getValue();

                int r = 0;

                // 写表头（可为空）
                if (!table.headers.isEmpty()) {
                    Row header = sheet.createRow(r++);
                    for (int c = 0; c < table.headers.size(); c++) {
                        Cell cell = header.createCell(c);
                        cell.setCellValue(table.headers.get(c));
                        cell.setCellStyle(headerStyle);
                    }
                    // 冻结首行
                    sheet.createFreezePane(0, 1);
                }

                // 写数据
                for (List<Object> rowData : table.rows) {
                    Row row = sheet.createRow(r++);
                    for (int c = 0; c < rowData.size(); c++) {
                        writeCell(row.createCell(c), rowData.get(c), dateStyle);
                    }
                }

                // 自动列宽（上限以避免巨宽列）
                int colCount = !table.headers.isEmpty() ? table.headers.size()
                        : table.rows.stream().mapToInt(List::size).max().orElse(0);
                for (int c = 0; c < colCount; c++) {
                    sheet.autoSizeColumn(c);
                    int width = sheet.getColumnWidth(c);
                    sheet.setColumnWidth(c, Math.min(width, 100 * 256)); // 最宽约 100 字符
                }
            }

            // 确保目录存在
            Path outPath = Path.of(outPathStr);
            if (outPath.getParent() != null) {
                Files.createDirectories(outPath.getParent());
            }
            wb.write(Files.newOutputStream(outPath));
        }
    }

    private static void writeCell(Cell cell, Object v, CellStyle dateStyle) {
        if (v == null) {
            cell.setBlank();
            return;
        }
        if (v instanceof Number) {
            cell.setCellValue(((Number) v).doubleValue());
        } else if (v instanceof Boolean) {
            cell.setCellValue((Boolean) v);
        } else if (v instanceof Date) {
            cell.setCellValue((Date) v);
            cell.setCellStyle(dateStyle);
        } else if (v instanceof Calendar) {
            cell.setCellValue((Calendar) v);
            cell.setCellStyle(dateStyle);
        } else if (v instanceof TemporalAccessor) {
            // Convert all date cells to strings.
            cell.setCellValue(v.toString());
        } else {
            cell.setCellValue(String.valueOf(v));
        }
    }

    /** Excel sheet 名限制处理（最长 31，不能包含 \ / ? * [ ] : ）*/
    private static String sanitizeSheetName(String name) {
        if (name == null || name.isEmpty()) name = "Sheet";
        String s = name.replaceAll("[\\\\/?*\\[\\]:]", "_");
        return s.length() > 31 ? s.substring(0, 31) : s;
    }

    // ===== 示例用法 =====
    public static void main(String[] args) throws IOException {
        // Sheet 1：用户
        List<String> userHeaders = Arrays.asList("ID", "Name", "Age", "Active");
        List<List<Object>> userRows = Arrays.asList(
                Arrays.asList(1, "Alice", 28, true),
                Arrays.asList(2, "Bob", 34, false),
                Arrays.asList(3, "Carol", 25, true)
        );

        // Sheet 2：订单
        List<String> orderHeaders = Arrays.asList("OrderID", "UserID", "Amount", "CreatedAt");
        List<List<Object>> orderRows = Arrays.asList(
                Arrays.asList("A001", 1, 199.5, new Date()),
                Arrays.asList("A002", 2, 89.0, new Date()),
                Arrays.asList("A003", 3, 520.0, new Date())
        );

        Map<String, DataTable> sheets = new LinkedHashMap<>();
        sheets.put("Users", DataTable.of(userHeaders, userRows));
        sheets.put("Orders", DataTable.of(orderHeaders, orderRows));

        writeXlsx(sheets, "output/multi_sheets_demo.xlsx");
        System.out.println("Excel 已生成：output/multi_sheets_demo.xlsx");
    }
}
