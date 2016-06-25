/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.model.naivebayes;

import java.io.File;
import java.util.Arrays;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

/**
 *
 * @author daniele
 */
public class Classifier {

//    private static final String PATH_MODEL = "/home/daniele/TrainEvalutate/dataset/myNaiveBayesModel";
//    private static final String PATH_HAM_DATA = "/home/daniele/TrainEvalutate/dataset/ham.txt";
//    private static final String PATH_SPAM_DATA = "/home/daniele/TrainEvalutate/dataset/spam.txt";
    private static final String PATH_MODEL = "dataset/myNaiveBayesModel";
    private static final String PATH_HAM_DATA = "dataset/ham.txt";
    private static final String PATH_SPAM_DATA = "dataset/spam.txt";

    private static SparkConf conf;
    private static JavaSparkContext jsc;

    private static JavaRDD<LabeledPoint> training;
    private static JavaRDD<LabeledPoint> test;

    public Classifier() {
        conf = new SparkConf()
                .setAppName("Train&Evalutate")
                .setMaster("local")
                //settiamo la directory in cui è presente SPARK
                .setSparkHome("/usr/local/src/spark-1.6.1-bin-hadoop2.6/");
        jsc = new JavaSparkContext(conf);
        File f = new File(PATH_MODEL);

        if (deleteModel(f)) {
            System.out.println("Model Successfully deleted");
        } else {
            System.out.println("Model not deleted");
        }

    }

    /**
     * Metodo che elabora i dataset presenti.
     * Sono presenti due file, uno contenente tutte le mail correttamente
     * etichettate come HAM, l'altro quelle etichettate come SPAM.      
     * @return Ritorna un RDD in cui sono presenti tutte le mail con le 
     * loro relative etichette.
     */
    public static JavaRDD<LabeledPoint> dataset() {
        final HashingTF tf = new HashingTF(10000);
        /*
        Importiamo il testo delle mail etichettate come HAM
        (testo inteso come Sender, Subject e Text).
         */
        JavaRDD<String> ham = jsc.textFile(PATH_HAM_DATA);
        /*
        Importiamo il testo delle mail etichettate come SPAM
        (testo inteso come Sender, Subject e Text).
         */
        JavaRDD<String> spam = jsc.textFile(PATH_SPAM_DATA);

        /*
        Effettua trasformazione testo di ogni singola mail in Hashing TF - Term 
        Frequency dopo aver rimosso gli spazi. La TF rappresenta il numero di 
        volte che il termine appare nel documento.
        Dopo la seguente trasformazione mapparemo le word in base alla label:
        1 - SPAM
        0 - HAM
         */
        JavaRDD<LabeledPoint> hamLabelledTF = ham.map(new Function<String, LabeledPoint>() {
            @Override
            public LabeledPoint call(String email) {
                return new LabeledPoint(0, tf.transform(Arrays.asList(email.toLowerCase().split(" "))));
            }
        });
        JavaRDD<LabeledPoint> spamLabelledTF = spam.map(new Function<String, LabeledPoint>() {
            @Override
            public LabeledPoint call(String email) {
                return new LabeledPoint(1, tf.transform(Arrays.asList(email.toLowerCase().split(" "))));
            }
        });

        /*
        Uniamo i due insiemi precedentemente ricavati ed elaborati.
         */
        JavaRDD<LabeledPoint> data = spamLabelledTF.union(hamLabelledTF);
        return data;
    }

     /**
     * Metodo che permette l'eliminazione della directory in cui è salvato
     * il modello precedentemente addestrato.
     * Il server prima di effettuare l'addestramento eliminerà tale
     * directory.
     * @return True se l'eliminazione viene eseguita con successo; 
     *         False se viene sollevata qualche eccezione.
     */
    public boolean deleteModel(File model) {
        if (model.isDirectory()) {
            String[] children = model.list();
            for (int i = 0; i < children.length; i++) {
                boolean success = deleteModel(new File(model, children[i]));
                if (!success) {
                    return false;
                }
            }
        }
        return model.delete();
    }

    /**     
     * Metodo che permette l'addestramento del modello sulla base di un 
     * dataset iniziale di mail correttamente etichettate come SPAM o HAM. 
     * I dataset sono stati importati da SpamAssassin.
     * @return True se l'addestramento viene eseguita con successo; 
     *         False se viene sollevata qualche eccezione.
     */
    public boolean train() {
        try {            
            JavaRDD<LabeledPoint> data = dataset();
            
            /*
            Il dataset iniziale viene suddiviso in maniera random. 
            Il 60 % del seguente dataset costituirà il Training Set il quale 
            verrà utilizzato per addestrare il modello;
            il restante 40 % invece costituisce il Test Set utile per la fase 
            di valutazione delle prestazioni del modello
            */
            JavaRDD<LabeledPoint>[] tmp = data.randomSplit(new double[]{0.6, 0.4}, 11L);
            //Training Set
            training = tmp[0]; 
            //Test Set
            test = tmp[1]; 

            //Addestramento del modello
            final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
            //Salvataggio del modello
            model.save(jsc.sc(), PATH_MODEL);

        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public static Double evalutate() {

        //Caricamento del modello precedentemente addestrato
        final NaiveBayesModel model = NaiveBayesModel.load(jsc.sc(), PATH_MODEL);
        
        /*
        Verrà utilizzato il Test Set per valutare le prestazioni del modello.
        Il Test Set contiene il 40 % delle mail già classificate presenti nel
        dataset iniziale. Verrà confrontata la previsione fornita dal modello
        con l'etichetta reale. 
        */
        JavaPairRDD<Double, Double> predictionAndLabel
                = test.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
                    }
                });
        /*
        L'accuratezza del modello è calcolata in base al numero di previsioni
        corrette fratto il numero totali di previsioni effettuate.
        */
        double accuracy = predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Double, Double> pl) {
                return pl._1().equals(pl._2());
            }
        }).count() / (double) test.count();
        return accuracy;
    }

}
