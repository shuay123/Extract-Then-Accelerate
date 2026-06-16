package util;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class CSVExporter {
    public static void exportToCSV(String[][] data, String csvFile) {
        try (FileWriter writer = new FileWriter(csvFile)) {
            // 写入表头
            writer.append("Name,Age,Country\n");
            // 写入数据
            writer.append("John Doe,30,USA\n");
            writer.append("Jane Smith,25,UK\n");
            writer.flush();
            System.out.println("CSV 文件导出成功！");
            } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void ExportTOCSV(String[] r, String csvFiledir, String csvFilename, boolean append) {
        String csvFile = csvFiledir + csvFilename;
        try (FileWriter writer = new FileWriter(csvFile, append)) {
            for (String row : r) {
                writer.append(row);
            }
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args, String csvFiledir, String csvFilename) {
        String[][] data = {
            {"Name", "Age", "Country"},
            {"John Doe", "30", "USA"},
            {"Jane Smith", "25", "UK"}
        };
        String csvFile = "dateset/output.csv";
        String csvFilePath = csvFiledir + csvFilename;
        exportToCSV(data, csvFilePath);
    }

}