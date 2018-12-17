package com.science.data.DL4JRecipe;

import java.util.Arrays;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DeepAutoEncoderExample {
	private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderExample.class);

	public static void main(String[] args)  throws Exception{
		//CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true); // GPU 사용
		//MNIST는 손글씨를 숫자를 모아놓은 데이터 셋 (1~10까지)
		final int numRows = 28;
		final int numColumns = 28;
		int seed = 123;
		int numSamples = MnistDataFetcher.NUM_EXAMPLES;
		int batchSize = 1000; //1회 학습 때 사용할 데이터 샘플 수
		int iterations = 1;
		int listenerFreq = iterations/5; //프로세스에서 비용함수의 값이 얼마나 자주 출력할 것인지 결정
		
		log.info("데이터 로드....");
		DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples, true);
		
		
		log.info("모델 생성....");
		//랜덤 seed 와 반복 횟수 설정 후 최적화 알고리즘을 선형 경사 하강법으로 설정
		//입력층 1개 4개 인코딩층, 4개 디코딩층 1개 출력층 total 10층
		MultiLayerConfiguration conf = new  NeuralNetConfiguration.Builder().seed(seed).iterations(iterations).optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).list(10)
				//역전파 알고리즘을 갖는 10개의 층 생성
				.layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(1, new RBM.Builder().nIn(1000).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(2, new RBM.Builder().nIn(500).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(3, new RBM.Builder().nIn(250).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(4, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(5, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(6, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(7, new RBM.Builder().nIn(250).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(8, new RBM.Builder().nIn(500).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(9, new OutputLayer.Builder().nIn(1000).nOut(numRows * numColumns).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.pretrain(true).backprop(true).build();
		
		//모델 초기화
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		
		//데이터 학습
		model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
		log.info("모델 학습....");
		while(iter.hasNext()) {
			DataSet next =iter.next();
			model.fit(new DataSet(next.getFeatures(), next.getFeatures()));
		}
	}
}
