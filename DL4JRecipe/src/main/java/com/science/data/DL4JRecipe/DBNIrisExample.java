package com.science.data.DL4JRecipe;

import java.util.Arrays;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DBNIrisExample {
	private static Logger log = LoggerFactory.getLogger(DBNIrisExample.class);

	public static void main(String[] args) throws Exception {

		Nd4j.MAX_SLICES_TO_PRINT = -1;
		Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

		final int numRows = 4;
		final int numColumns = 1; // 1차원
		int outputNum = 3;
		int numSamples = 150; // 전체 데이터 수
		int batchSize = 150; // 한번의 배치마다 학습실킬 데이터 수
		int iterations = 5;
		int splitTratinNum = (int) (batchSize * .8); // 트레이닝 셋과 테스트 셋 분할 80% 트레이닝, 나머지 테스트
		int seed = 123;
		int listenerFreq = 1; // 프로세스를 진행하면서 비용함수의 값을 얼마나 자주 로그로 볼것인지 - 1 은 반복할때마다 로그 출력

		log.info("데이터 로드....");
		DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);

		// 데이터 정규화
		DataSet next = iter.next();
		next.normalizeZeroMeanZeroUnitVariance();

		log.info("데이터 분할....");
		// 데이터 분할 시 무작위 선택을위해 seed 값 사용
		// 학습효과를 높이기 위해 수치적 안저어성을 적용하여 설정
		SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTratinNum, new Random(seed));
		DataSet train = testAndTrain.getTrain();
		DataSet test = testAndTrain.getTest();
		Nd4j.ENFORCE_NUMERICAL_STABILITY = true; // 학습효과를 높이기 위해 수치적 안저어성을 적용하여 설정

		// 모델 생성
		/*
		 * seed 를 통해 가중치 초기화 iterations 예측과 분류의 학습의 반복 횟수 gradients(경사도)를 계산하기 위해 back
		 * propagation(역전파) 알고리즘 사용 list 신경망 층 수에 대한 매개변수 (입력층 제외)
		 */
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
				.learningRate(1e-6f).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).l1(1e-1)
				.regularization(true).l2(2e-4).useDropConnect(true).list(2)
				/*
				 * 신경망의 첫번째 층 설정 소스 0은 층의 인덱스 번호 k 대조산발을 실행하는 횟수(학습에 사용하는 심층 신경망에서 가중치와 오차의
				 * 기울기(grandients)를 구한느 알고리즘 RBM.VisibleUnit.GAUSSIAN 사용으로 Iris의 부동소수값 데이터 모델이
				 * 연속된 수를 처리할수 있도록함 Updater.ADAGRAD 학습률을 최적화하기 위해 사용
				 */
				.layer(0,
						new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN).nIn(numRows * numColumns)
								.nOut(3).weightInit(WeightInit.XAVIER).k(1).activation("relu")
								.lossFunction(LossFunctions.LossFunction.RMSE_XENT).updater(Updater.ADAGRAD)
								.dropOut(0.5).build())
				/*
				 * 두번째 레이어 설정
				 */
				.layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(3).nOut(outputNum).activation("softmax").build())
				.build();

		// 모델 생성
		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
		log.info("모델 학습....");
		model.fit(train);

		log.info("추정된 가중치....");
		for (org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
			INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
			log.info("Weights : " + w);
		}

		// 모델 평가
		log.info("모델 평가....");
		Evaluation eval = new Evaluation(outputNum);
		INDArray output = model.output(test.getFeatureMatrix());
		for (int i = 0; i < output.rows(); i++) {
			String actual = test.getLabels().getRow(i).toString().trim();
			String predicted = output.getRow(i).toString().trim();
			log.info("actual " + actual + " vs predicted " + predicted);
		}
		eval.eval(test.getLabels(), output);
		log.info(eval.stats());
	}
}
