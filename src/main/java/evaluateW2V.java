import java.util.Collection;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
/**
 * Created by niless on 4/19/16.
 */
public class evaluateW2V {
    private static Logger log = LoggerFactory.getLogger(applyW2V.class);

    public static void main(String[] args) throws Exception {

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File("/Users/efsun/Documents/GWU/Results/writingAll.txt"));
        WeightLookupTable weightLookupTable = wordVectors.lookupTable();
       // double[] wordVector = wordVectors.getWordVector("myword");

        log.info("Evaluate model...");
        double sim = wordVectors.similarity("gym", "basketball");
        log.info("Similarity between gym and basketball" + sim);
        Collection<String> similar = wordVectors.wordsNearest("bible", 10);
        log.info("Similar words to bible " + similar);
    }
}
