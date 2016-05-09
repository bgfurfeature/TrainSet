package train

import org.apache.spark.mllib.classification.{NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by C.J.YOU on 2016/5/9.
  */
object DigitRecognizer {

  // 特征标准化
  def standardFeatures(input:RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val vectors = input.map(lp => lp.features)
    val  scaler = new StandardScaler(withMean = false,withStd = true).fit(vectors)
    val scaledData = input.map(lp => LabeledPoint(lp.label,
      scaler.transform(lp.features)
    ))
    scaledData
  }

  // NB diff lambda function
  def trainNBWithParams(input:RDD[LabeledPoint],lambda:Double,modelType:String) = {
    val nbModel = new NaiveBayes()
      .setLambda(lambda)
      .setModelType(modelType)
      .run(input)
    nbModel
  }

  // dt
  def trainDTWithParams(input:RDD[LabeledPoint],maxDepth:Int,impurity: Impurity,numberClass:Int) = {
    DecisionTree.train(input,Algo.Classification,impurity,maxDepth,numberClass)
  }

  // svm
  def trainSVMWithParams(input:RDD[LabeledPoint],regParam:Double,numIteration:Int,updater:Updater,stepSize:Double) = {
    val SVMModel = new SVMWithSGD
    SVMModel.optimizer.setNumIterations(numIteration)
      .setRegParam(regParam)
      .setStepSize(stepSize)
      .setUpdater(updater)
    SVMModel.run(input)
  }
}
