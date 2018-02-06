package com.ctbc.Weka_test;

import javax.swing.JFrame;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

// 資料源： http://repository.seasr.org/Datasets/UCI/arff/
public class Weka {

	public static Instance setAnNewAnimal(double weight, double[] features) {
		Instance myUnicorn = new DenseInstance(weight, features);
		return myUnicorn;
	}

	public static void main(String[] args) {
		try {
			DataSource source = new DataSource(System.getProperty("user.dir") + "\\src\\main\\resources\\資料集\\zoo.arff");
			Instances data = source.getDataSet();

			System.out.println(data.numInstances() + " instances loaded. ");
			System.out.println("==============================");
			System.out.println(data.toString());
			System.out.println("==============================");

			Remove rm = new Remove();
			String[] opts = new String[] { "-R", "1" };

			rm.setOptions(opts);
			rm.setInputFormat(data);
			data = Filter.useFilter(data, rm);
			System.out.println(data.toString());
			System.out.println("==============================");

			InfoGainAttributeEval eval = new InfoGainAttributeEval();
			Ranker search = new Ranker();

			AttributeSelection attSelect = new AttributeSelection();
			attSelect.setEvaluator(eval);
			attSelect.setSearch(search);
			attSelect.SelectAttributes(data);

			int[] indices = attSelect.selectedAttributes();
			System.out.println(Utils.arrayToString(indices));

			/*
			 * Build a decision tree
			 */
			String[] options = new String[] { "-U" }; // Use unpruned tree.
			J48 tree = new J48();
			tree.setOptions(options);
			tree.buildClassifier(data);
			System.out.println(tree);

			/*
			 * Visualize decision tree
			 */
			TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2());
			JFrame frame = new javax.swing.JFrame("Tree Visualizer");
			frame.setSize(800, 500);
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			frame.getContentPane().add(tv);
			frame.setVisible(true);
			tv.fitToScreen();

			/*
			 * 建立新物種特徵
			 */
			double weight = 1.0;
			double[] features = new double[] {
							1.0, 0.0, 0.0, 1.0, 0.0,
							0.0, 0.0, 0.0, 1.0, 1.0,
							1.0, 0.0, 4.0, 1.0, 1.0,
							0.0
			};
			Instance myUnicorn = Weka.setAnNewAnimal(weight, features);
			myUnicorn.setDataset(data);
			double result = tree.classifyInstance(myUnicorn);
			System.out.println("===========================================");
			System.out.println("result : " + result);
			System.out.println("class : " + data.classAttribute().value((int) result));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
