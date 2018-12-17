package word2vec.chap8.science.data;

import java.util.ArrayList;
import java.util.Collection;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Word2VecRawTestExample {
	private static Logger log = LoggerFactory.getLogger(Word2VecRawTestExample.class);

	public static void main(String[] args) throws Exception {
		String filePath = "data/raw_sentences.txt";

		log.info("문장 로드 & 벡터화....");
		// 각줄 앞뒤 공백 제거
		SentenceIterator iter = UimaSentenceIterator.createWithPath(filePath);
		// 공백을 기준으로 각 줄의 단어 토큰화
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());
		InMemoryLookupCache cache = new InMemoryLookupCache();
		WeightLookupTable table = new InMemoryLookupTable.Builder().vectorLength(100).useAdaGrad(false).cache(cache)
				.lr(0.025f).build();

		log.info("모델 생성....");
		Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5).iterations(1).layerSize(5).lookupTable(table)
				.stopWords(new ArrayList<String>()).vocabCache(cache).seed(65).windowSize(5).iterate(iter)
				.tokenizerFactory(t).build();
		/*
		minWordFrequency - 최소 출현 빈도
		iterations 한 묶음의 데이터에 대해 신경망이 학습을 진행해 계수를 업데이트 하는 과정을 몇번 반복할 것인가
		razerSize 단어 벡터의 피쳐수 또는 피쳐의 차원
		tokenizer 데이터로부터 단어들 추출
		*/
		
		log.info("Word2Vec 모델 학습....");
		vec.fit();
		
		log.info("단어 벡터를 텍스트 파일로 저장....");
		//단어 저장
		WordVectorSerializer.writeWordVectors(vec, "word2vec.txt");
		
		log.info("근접 단어:");
		Collection<String> lst = vec.wordsNearest("man", 5);
		System.out.println(lst);
		double cosSim = vec.similarity("cruise", "voyage");
		System.out.println(cosSim);

	}

}
