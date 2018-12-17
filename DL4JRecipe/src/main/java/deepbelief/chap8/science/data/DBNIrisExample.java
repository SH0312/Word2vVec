package deepbelief.chap8.science.data;

import java.util.Arrays;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
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
import org.deeplearning4j.nn.api.Layer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DBNIrisExample {
	private static Logger log = LoggerFactory.getLogger(DBNIrisExample.class);

	public static void main(String[] args) throws Exception {
		Nd4j.MAX_SLICES_TO_PRINT = -1;
		Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

		final int numRows = 4;
		final int numColumns = 1;
		int outputNum = 3;
		int numSamples = 150;
		int batchSize = 150;
		int iterations = 5;
		int splitTrainNum = (int) (batchSize * .8);
		int seed = 123;
		int listenerFreq = 1;

		log.info("데이터 로드....");
		DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
		DataSet next = iter.next();
		next.normalizeZeroMeanZeroUnitVariance();

		log.info("데이터 분할....");
		SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
		DataSet train = testAndTrain.getTrain();
		DataSet test = testAndTrain.getTest();
		Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

		log.info("모델 생성....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
				.learningRate(1e-6f).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT).l1(1e-1)
				.regularization(true).l2(2e-4).useDropConnect(true).list(2)
				.layer(0,
						new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN).nIn(numRows * numColumns)
								.nOut(3).weightInit(WeightInit.XAVIER).k(1).activation("relu")
								.lossFunction(LossFunctions.LossFunction.RMSE_XENT).updater(Updater.ADAGRAD)
								.dropOut(0.5).build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(3).nOut(outputNum)
						.activation("softmax").build())
				.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		
		model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
		
		
		
		log.info("모델 학습....");
		model.fit(train);
		
		log.info("추정된 가중치....");
		for(Layer layer : model.getLayers()) {
			INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
			log.info("Weights: " + w);
		}
		
		log.info("모델 평가....");
		Evaluation eval = new Evaluation(outputNum);
		INDArray output = model.output(test.getFeatureMatrix());
		for(int i = 0 ; i < output.rows(); i++) {
			String actual = test.getLabels().getRow(i).toString().trim();
			String predicted = output.getRow(i).toString().trim();
			
			log.info("actual " + actual + " vs predicted " + predicted);
		}
		
		eval.eval(test.getLabels(), output);
		log.info(eval.stats());
	}
}
