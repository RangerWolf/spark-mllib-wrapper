package com.trendmicro.skyaid.utils.common;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.google.common.collect.Lists;

public class MLlibRandomForestWrapperDemo implements java.io.Serializable {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1352215886461017214L;

	public static void main(String[] args) throws IOException {
		String taskName = "SparkWrapper Test ";
		SparkSession ss = SparkConnUtils.getSession(taskName, true);
		
		// 首先读iris 数据
		// 因为是从本地读取Sample数据，所以比较麻烦一些~
		List<String> lines = FileUtils.readLines(new File("E:\\DataSet\\iris_data.txt"), "UTF-8");
		List<Row> data = Lists.newArrayList();
		String[] headers = lines.get(0).split(",");
		for(String line : lines.subList(1, lines.size())) {
			// 前面几个都是double
			String[] cells = line.split(",");
			
			Object[] values = new Object[cells.length];
			for(int i = 0; i < cells.length - 1; i++) {
				values[i] = Double.parseDouble(cells[i]);
			}
			values[cells.length - 1] = cells[cells.length - 1];
			data.add(RowFactory.create(values));
		}
		
		// 创建Dataset
		StructField[] fields = new StructField[headers.length];
		for(int i = 0; i < headers.length - 1; i++) {
			fields[i] = new StructField(headers[i], DataTypes.DoubleType, false, Metadata.empty());
		}
		fields[headers.length - 1] = new StructField(headers[headers.length - 1], DataTypes.StringType, false, Metadata.empty());
		StructType schema = new StructType(fields);
		Dataset<Row> df = ss.createDataFrame(data, schema);
		df.show();
		
		
		MLlibRandomForestWrapper wrapper = new MLlibRandomForestWrapper(df, "classes");
		
		wrapper.trainClassifier(0.3, 30, 3, 32, 12345);
		
		wrapper.predictLabel();
		
		wrapper.matrixReport();
	}
}
