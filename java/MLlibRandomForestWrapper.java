package com.trendmicro.skyaid.utils.common;

import java.beans.Transient;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.spark_project.guava.collect.Lists;

import com.google.gson.Gson;

import jersey.repackaged.com.google.common.collect.Sets;
import scala.Tuple2;

/**
 * this is just to make spark mllib randorm forest use easy
 * @author wenjun_yang
 *
 */
public class MLlibRandomForestWrapper implements java.io.Serializable{

	private static final long serialVersionUID = 3295321497228560277L;
	
	private Dataset<Row> ds;
	private String labelColName;
	private String[] featureColNames = null;
	private JavaRDD<LabeledPoint>[] splits;
	private RandomForestModel model;
	private JavaRDD<Tuple2<Object, Object>> predictionAndLabels;
	private int numClasses = 0;
	
	private static final String STR_INDEX_POSTFIX = "_strIndex";
	
	public MLlibRandomForestWrapper(Dataset<Row> ds, String labelColName) {
		this.ds = ds;
		this.labelColName = labelColName;
	}
	
	public MLlibRandomForestWrapper(Dataset<Row> ds, String labelColName, String[] featureColNames) {
		this.ds = ds;
		this.labelColName = labelColName;
		this.featureColNames = featureColNames;
	}
	
	public Dataset<Row> strLabelToIndex() {
		// 首先是StringIndexer
		StringIndexer indexer = new StringIndexer()
		  .setInputCol(this.labelColName)
		  .setOutputCol(this.labelColName + STR_INDEX_POSTFIX);
		Dataset<Row> indexed = indexer.fit(ds).transform(ds);
		
		return indexed;
	}
	
	
	public Integer getLabelIndex(Dataset<Row> indexed) {
		// 找出哪些列是feature列， 以及他们的列下标
		String[] allColNames = indexed.columns();
		
		
		int labelIdx = 0;
		// 找出label在第几列
		for(int i = 0; i < allColNames.length; i++) {
			if(allColNames[i].equals(this.labelColName + STR_INDEX_POSTFIX)) {
				labelIdx = i;
				break;
			}
		}
		
		return labelIdx;
	}
	
	public List<Integer> getFeatureIndex(Dataset<Row> indexed) {
		// 找出哪些列是feature列， 以及他们的列下标
		String[] allColNames = indexed.columns();
		
		// 去除label列
		Set<String> featureColNames = Sets.newHashSet(allColNames);
		featureColNames.remove(this.labelColName);
		featureColNames.remove(this.labelColName + STR_INDEX_POSTFIX);
		
		List<Integer> featureColIdxLst = Lists.newArrayList();
		if(featureColNames != null) {
			featureColNames = Sets.newHashSet(featureColNames);
		}
		
		for(int i = 0; i < allColNames.length; i++) {
			if(featureColNames.contains(allColNames[i])) {
				featureColIdxLst.add(i);
			}
		}
		
		return featureColIdxLst; 
	}
	
	public JavaRDD<LabeledPoint> mapToLabeledPointData(Dataset<Row> indexed) {
		final int finalLabelIdx = getLabelIndex(indexed);
		
		List<Integer> featureColIdxLst = getFeatureIndex(indexed);
		System.out.println(new Gson().toJson(indexed.columns()));
		System.out.println(new Gson().toJson(featureColIdxLst));
		// 开始转换, Assume 每一个feature都是Double类型的数值. Label 经过前面的stringIndexer 已经是double类型了
		// 之后再过改进
		JavaRDD<LabeledPoint> rowRDD = indexed.toJavaRDD().map(new Function<Row, LabeledPoint>() {
			@Override
			public LabeledPoint call(Row row) throws Exception {
				double[] features = new double[featureColIdxLst.size()];
				for(int i = 0; i < featureColIdxLst.size(); i++) {
					features[i] = row.getDouble(i);
				}
				System.out.println(new Gson().toJson(features));
				LabeledPoint point = new LabeledPoint(row.getDouble(finalLabelIdx), Vectors.dense(features));
				return point;
			}
		});
		
		return rowRDD;	
	}

	public void predictLabel() {
		JavaRDD<Tuple2<Object, Object>> predictionAndLabels = this.splits[1].map(
		  new Function<LabeledPoint, Tuple2<Object, Object>>() {
		    public Tuple2<Object, Object> call(LabeledPoint p) {
		      Double prediction = model.predict(p.features());
		      return new Tuple2<Object, Object>(prediction, p.label());
		    }
		  }
		);
		
		this.predictionAndLabels = predictionAndLabels;
	}
	
	public void trainClassifier( double testSize, int numTrees,
			int maxDepth, int maxBins, int seed) {
		
		Dataset<Row> indexed = strLabelToIndex();
		
		JavaRDD<LabeledPoint> pointData = mapToLabeledPointData(indexed);
		
		splits = pointData.randomSplit(new double[]{1-testSize, testSize});
		JavaRDD<LabeledPoint> trainingData = splits[0];
		JavaRDD<LabeledPoint> testData = splits[1];
		
		String featureSubsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "gini";
		
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
		
		// 开始自动识别有多少个class
		this.numClasses = indexed.select(this.labelColName).dropDuplicates().collectAsList().size();
		
		// 开始自动判断每个feature是什么类型
		List<Integer> featureIdxList = getFeatureIndex(indexed);
		int pointDataFeatureIdx = 0;		// 因为indexed经过转换成RDD之后，其feature的下标完全有可能已经被修改了
		String[] allColNames = indexed.columns();
		long totalRowCnt = indexed.count();
		for(int i = 0; i < featureIdxList.size(); i++) {
			int curSize = indexed.select(allColNames[i]).dropDuplicates().collectAsList().size();
			if( totalRowCnt / curSize > 10) {
				// 相当于平均每个类别有10条记录， 认为该feature非连续性的feature
				categoricalFeaturesInfo.put(pointDataFeatureIdx, curSize);
			} else {
				// 如果是连续的就算了，直接pass
			}
			
			pointDataFeatureIdx++;
		}
		
		final RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses,
				  categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
				  seed);
		
		this.model = model;
	}
	
	/**
	 * 返回预测结果
	 * @return
	 */
	public JavaRDD<Tuple2<Object, Object>> getPredResult() {
		// TODO: 目前这种结果，都是Double类型的，因为feature、label都已经经过了StringIndexer的转换. 需要将其变成之前的数据，方便查看
		return predictionAndLabels;
	}
	
	
	/**
	 * 优化之后的matrix report
	 */
	public void matrixReport() {
		if(this.numClasses > 2) {
			mulClsMatrixReport();
		} else {
			System.out.println("Binary classification matrix report not implemented");
		}
	}
	
	private void mulClsMatrixReport() {
		MulticlassMetrics mulClsMetrics = new MulticlassMetrics(predictionAndLabels.rdd());
		System.out.printf("Accuracy: %.4f\n", mulClsMetrics.accuracy());
		System.out.println("Confusion Matrix:");
		System.out.println(mulClsMetrics.confusionMatrix());
	}
	
}
