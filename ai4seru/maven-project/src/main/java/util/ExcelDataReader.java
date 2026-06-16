package util;

import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class ExcelDataReader {

    public List<List<Object>> readExcelSheet(String filePath, String sheetName) throws Exception {
        List<List<Object>> data = new ArrayList<>();

        try (InputStream inputStream = getClass().getClassLoader().getResourceAsStream(filePath)) {
            if (inputStream == null) {
                throw new Exception("文件 '" + filePath + "' 在 resources 目录中不存在");
            }

            Workbook workbook = createWorkbook(filePath, inputStream);
            Sheet sheet = workbook.getSheet(sheetName);
            if (sheet == null) {
                throw new Exception("工作表 '" + sheetName + "' 不存在");
            }

            for (Row row : sheet) {
                List<Object> rowData = new ArrayList<>();
                for (org.apache.poi.ss.usermodel.Cell cell : row) {   // ← 用全限定名
                    rowData.add(getCellValue(cell));
                }
                data.add(rowData);
            }
            workbook.close();
        }
        return data;
    }

    private Workbook createWorkbook(String filePath, InputStream inputStream) throws Exception {
        if (filePath.endsWith(".xlsx")) return new XSSFWorkbook(inputStream);
        return new HSSFWorkbook(inputStream);
    }

    private Object getCellValue(org.apache.poi.ss.usermodel.Cell cell) { // ← 用全限定名
        switch (cell.getCellType()) {
            case NUMERIC: return cell.getNumericCellValue();
            case STRING:  return cell.getStringCellValue();
            default:      return "";
        }
    }
}
