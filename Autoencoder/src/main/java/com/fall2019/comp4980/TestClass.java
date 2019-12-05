/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.fall2019.comp4980;

import static com.google.common.collect.Iterables.skip;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import javax.imageio.ImageIO;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.PrintWriter;
import java.util.HashMap;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * @author Hemanth and Zach
 */
public class TestClass {
    
    public static volatile boolean skip = false; 
    public static ComputationGraph tsne_ae2;
    private static Random rn = new Random();
    private static ArrayList<INDArray> array_images = new ArrayList<INDArray>();
    private static ArrayList<INDArray> embedding_activations = new ArrayList<INDArray>();
    private static ArrayList<INDArray> testing_activations = new ArrayList<INDArray>();
    private static INDArray embedding_v;

    private static File filename = new File("output.csv");
    
    private static Map<String, INDArray> activations;
    private static Map<String, INDArray> biometric_activations;

    public static void main(String[] args) throws Exception
    {
        Img2INDArray.grayscale();
        tsne_ae2 = nn_init2();
        boolean train2 = false;
        try{
            tsne_ae2 = ComputationGraph.load(new File("CNN_autoencoder.zip"),true);
            System.out.println("TSNE2 Existing model loaded");
        }
        catch(Exception e){
            System.out.println("TSNE2 Starting from scratch");
            tsne_ae2 = nn_init2();
            train2 = true;
        }
        
        if(train2){train_autoencoder(10);}

        test_autoencoder();

    }
    private static void load_images(String dir, int scaleX, int scaleY ) throws Exception
       {
        File directory;
        directory  = new File( dir );
        File[] fList = directory.listFiles();
            
        for (File file : fList){
       // System.out.println( file.getPath() );
        BufferedImage img = ImageIO.read( new File( file.getPath() ) );
        INDArray v_in = Img2INDArray.load_image(file.getPath(),scaleX,scaleY,0,0, 0.00, false);
        array_images.add(v_in);
        }
    }   
 
/**Training the autoencoder part
 * Train Hemanth and Zacs images only.   
    
*/    
     private static void train_autoencoder(int epochs) throws Exception{

        INDArray[] INPs = new INDArray[1];
        INDArray[] OUTs = new INDArray[1];
        INDArray result_out;

        for (int epoch =0; epoch< epochs; epoch++)
        {
            for (int j =0; j<2; j++){
                 File readfile = new File("Images/Train/"+j);
                 File[] listOfFiles = readfile.listFiles();
                 for (int i =0; i<listOfFiles.length;i++){
                        result_out= Nd4j.zeros(new int[]{1,100*100});
                        result_out.putScalar(0,j,1);
                        if(listOfFiles[i].isFile() && listOfFiles[i].getName().contains(".jpg")){
                        Img2INDArray.grayscale();
                        INPs[0] = Img2INDArray.load_image("Images/Train/"+j+"/"+listOfFiles[i].getName(),100 , 100, 0, 0, 0.0, false);
                        OUTs[0] = result_out;
                        tsne_ae2.fit(INPs, OUTs);
                        System.out.println( "SCORE=" + tsne_ae2.score() + "\tIMG#" + i + "\tEPOCH=" + epoch );}
                 }
            }
        }
            System.out.println("SAVING ");
            try {
                    tsne_ae2.save( new File("CNN_autoencoder.zip"), true);
                } catch(Exception e) {
                }
        }
     
     /**
      * Biometrics has all the embedding vectors that need to be checked.
      * @throws Exception 
      */
     private static void biometrics() throws Exception{

          PrintWriter pw = new PrintWriter(filename);
          INDArray[] INPs = new INDArray[1];
          for (int j=0; j<2; j++){
              File readfile = new File("Images/Train/"+j);
              File[] listOfFiles = readfile.listFiles();
              for (int i=0; i<listOfFiles.length; i++){
                  if(listOfFiles[i].isFile() && listOfFiles[i].getName().contains(".jpg")){
                      Img2INDArray.grayscale();
                      INPs[0] = Img2INDArray.load_image("Images/Train/"+j+"/"+listOfFiles[i].getName(),100 , 100, 0, 0, 0.0, false);
                      biometric_activations = tsne_ae2.feedForward(INPs,false);
                      embedding_v = biometric_activations.get("ENCODER_H2");
                      embedding_activations.add(embedding_v );
                     
                     // System.out.println(listOfFiles[i].getName() + "==>" + embedding_v);
                      pw.println(listOfFiles[i].getName()+","+embedding_v.toString());
                  }
              }
          }
          pw.close();
      }

    @SuppressWarnings("empty-statement")
    
     private static void test_autoencoder() throws Exception{
         File folder;
        folder  = new File("Images/Test/0");
        
        File[] fList = folder.listFiles();
         biometrics();
         double dist[] = new double[20];
         INDArray[] INPs = new INDArray[1];
         double euclidean_test_sum = 0.0;
        
         try(FileWriter fw = new FileWriter("output.csv",true);
                 BufferedWriter bw = new BufferedWriter(fw);
                 PrintWriter out = new PrintWriter(bw);)
                 {   
                     // Folder 0
                    for(int i=0; i<1;i++){
                        array_images.clear();
                        load_images("Images/Test/"+i,100,100);
                
                    for (INDArray img : array_images){
                        INPs[0] = img;
                        activations = tsne_ae2.feedForward(INPs,false);

                        INDArray test_embedding_v = activations.get("ENCODER_H2");
                        //System.out.println(" Test result"+ test_embedding_v);
                        testing_activations.add(test_embedding_v);
                        out.append("Unknown"+","+test_embedding_v.toString());
                        out.print("\n");
                        
                        euclidean_test_sum = getEuclidean(test_embedding_v);
                    
                    }
                    
                     double hmin = Double.MAX_VALUE;
                     double zmin = Double.MAX_VALUE;
                     
                     for (int k=0; k< fList.length;k++)
                        {
                        
                        for (int j =0; j<10; j++){
                            double euclidean = Transforms.euclideanDistance(testing_activations.get(k), embedding_activations.get(j));
                       
                                 if (euclidean < hmin){
                                     hmin = euclidean;
                                     }
                                 }
                        for (int j=10; j<20;j++){
                            double euclidean = Transforms.euclideanDistance(testing_activations.get(k), embedding_activations.get(j));
                            if (euclidean < zmin){
                                zmin = euclidean;
                            }
                        }
                      
                             
                        if (hmin < zmin){
                            dist[k] =hmin;
                            hmin = Double.MAX_VALUE;
                            zmin = Double.MAX_VALUE;
                            System.out.println(fList[k].getPath()+" ==> "+"Closest distance to Hemanth ==>"+ dist[k]+ " Uncertainity ==> "+ (dist[k]/euclidean_test_sum)*100);
                        }
                        else{
                            dist[k] =zmin;
                            hmin = Double.MAX_VALUE;
                            zmin = Double.MAX_VALUE;
                            System.out.println(fList[k].getPath()+" ==> "+"Closest distance to zach ==>"+ dist[k] + " Uncertainity ==> "+ (dist[k]/euclidean_test_sum)*100);
                        }
     
                        }
                 
                     System.out.println("\n");
                   } 

        }
        catch (IOException e){
        }
    }

     private static double getEuclidean(INDArray testRow) {
  
        double sum = 0.0;
        for(INDArray bio : embedding_activations )
        {
                double euclidean = Transforms.euclideanDistance(testRow, bio);
                sum += euclidean;
        }
        return sum;
        }
   
 /* The Autoencoder Part */   
    private static ComputationGraph nn_init2()
        {
            // Some hyperparameters (variable) declarations here.
        double learningRate = 0.0008;   // Learning rate

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)      
            .activation(Activation.ELU)         
            .updater(new Adam(learningRate))    
            //.l2(0.00001)                      
            .graphBuilder()
   
            .addInputs("vector_in")           

            .addLayer("INPUT_I1", new ConvolutionLayer.Builder()  
                .kernelSize(2,2)
                .stride(1,1)
                .nIn(1)                                     
                .nOut(16)                                    
                .build(), "vector_in")                      

           .addLayer("SUB1", new SubsamplingLayer.Builder(PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)   
                .build(), "INPUT_I1")
                    
                
            .addLayer("CONV2", new ConvolutionLayer.Builder()  
                .kernelSize(2,2)
                .stride(1,1)
                .nIn(16)                                     
                .nOut(32)                                    
                .build(), "SUB1")                      

           .addLayer("SUB2", new SubsamplingLayer.Builder(PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)   
                .build(), "CONV2")
                
                
           .addLayer("CONV3", new ConvolutionLayer.Builder()  
                .kernelSize(2,2)
                .stride(1,1)
                .nIn(32)                                     
                .nOut(64)                                    
                .build(), "SUB2")                      

           .addLayer("SUB3", new SubsamplingLayer.Builder(PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)   
                .build(), "CONV3")
                            
            
                /** The autoencoder part begins. Compress down to 25 nodes */
                
           .addLayer("ENCODER_H1", new DenseLayer.Builder()
                //.nIn(1)       // 18432  // 7743
                .nOut(3500)
                .build(), "SUB3")
                
            .addLayer("ENCODER_H2", new DenseLayer.Builder()
                .nIn(3500)
                .nOut(2500)
                .build(), "ENCODER_H1")

            .addLayer("DECODER_H6", new DenseLayer.Builder()
                .nIn(2500)
                .nOut(3500)
                .build(), "ENCODER_H2")
                
                
           .addLayer("OUTPUT_O1", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)  
                .activation( Activation.IDENTITY )                                         
                .nIn( 3500 )                                                                
                .nOut( 10000 )                                                         
                .build(), "DECODER_H6")

            .setOutputs("OUTPUT_O1")
            .setInputTypes(InputType.convolutional(100, 100, 1))    // height,width,channels
                
                
            .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();


         return net;
        }
}



















































