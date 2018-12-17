package com.science.data.DL4JRecipe;

import java.util.ArrayList;
import java.util.Collection;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

public class Word2VecRawTextExample {
	private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

	public static void main(String[] args) throws Exception {
		String filePath = "data/raw_sentences.txt";
		log.info("문장 로드 & 백터화 ...");
		SentenceIterator iter = UimaSentenceIterator.createWithPath(filePath);

		// wor2vector 은 문장보다는 토큰을 사용하기에 토큰화 작업
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		InMemoryLookupCache cache = new InMemoryLookupCache();
		WeightLookupTable table = new InMemoryLookupTable.Builder().vectorLength(100).useAdaGrad(false).cache(cache)
				.lr(0.025f).build();

		// 신경망 구성
		log.info("모델 생성....");
		Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5).iterations(1).layerSize(100).lookupTable(table)
				.stopWords(new ArrayList<String>()).vocabCache(cache).seed(42).windowSize(5).iterate(iter)
				.tokenizerFactory(t).build();
		//minWordFrequency 말뭉치에 나타나는 단어의 최소 출현 빈도 - 소스상에서 5번 이상나타나야 학습함.
		
		
		// 모델 학습
		log.info("Word2Vec 모델 학습....");
		vec.fit();
		
		log.info("단어 벡터를 텍스트 파일로 저장....");
		WordVectorSerializer.writeWordVectors(vec, "data/word2vec.txt");
		
		//피쳐 벡터 품질 평가
		//vec.wordsNearest - 신경망에 의해 의미상 유사한 단어로 군집화된 단어 목록을 제공 - 소스상에서 man 이란 단어와 유사한 단어 5개
		//vec.similarity   - 코사인 유사도 정보 출력 - 소스상에서 두단어의 코사인 유사도 정보 출력
		log.info("근접 단어 : ");
		Collection<String> lst = vec.wordsNearest("man", 5);
		System.out.println(lst);
		double cosSim = vec.similarity("cruise", "voyage");
		System.out.println(cosSim);
		
	}
}
