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
 * @author John
 */
public class TestClass {
    
    public static volatile boolean skip = false; 
    public static ComputationGraph cnn_ae;
    public static ComputationGraph tsne_ae2;
    private static Random rn = new Random();
    private static ArrayList<INDArray> array_images = new ArrayList<INDArray>();
    private static ArrayList<INDArray> embedding_activations = new ArrayList<INDArray>();
    private static INDArray embedding_v;
    private static File filename = new File("output.csv");
    
    private static Map<String, INDArray> activations;
    private static Map<String, INDArray> biometric_activations;

    public static void main(String[] args) throws Exception
    {
        Img2INDArray.grayscale();
        
        cnn_ae = nn_init(); 
        boolean train =false;
        try{
            cnn_ae = ComputationGraph.load(new File("CNN909.zip") , true);
            System.out.println("Existing model loaded");
        }
        catch(Exception e){
            System.out.println("Starting from scratch");
            cnn_ae = nn_init();
            train = true;
        }
        if(train){train(50);}
        test();

    }
         private static void load_images(String dir, int scaleX, int scaleY ) throws Exception
            {
            File directory;
            directory  = new File( dir );
            File[] fList = directory.listFiles();
            
            for (File file : fList){
            System.out.println( file.getPath() );
            BufferedImage img = ImageIO.read( new File( file.getPath() ) );
            INDArray v_in = Img2INDArray.load_image(file.getPath(),scaleX,scaleY,0,0, 0.00, false);
            array_images.add(v_in);
       }
    }
 
/*Training the CNN train part */  
    private static void train(int epochs) throws Exception
    {
        INDArray[] INPs = new INDArray[1];
        INDArray[] OUTs = new INDArray[1];
        INDArray result_out;
        int offsetX     = 0;
        int offsetY     = 0;
        for (int epoch =0; epoch< epochs; epoch++)
        {
            for (int j =0; j<10; j++){

                 File readfile = new File("Images/Train/"+j);
                 File[] listOfFiles = readfile.listFiles();

                 for (int i =0; i<listOfFiles.length;i++){
      
                        result_out= Nd4j.zeros(new int[]{1,10});
                        result_out.putScalar(0,j,1);
                        if(listOfFiles[i].isFile() && listOfFiles[i].getName().contains(".jpg")){
                        Img2INDArray.grayscale();
                        INPs[0] = Img2INDArray.load_image("Images/Train/"+j+"/"+listOfFiles[i].getName(),100 , 100, -10 + rn.nextInt(21), -10 + rn.nextInt(21), 0.0, false);
           
                        OUTs[0] = result_out;
           
                        cnn_ae.fit(INPs, OUTs);
                        System.out.println( "score=" + cnn_ae.score() + "\tepoch=" + epoch + "\t" + OUTs[0] );
                        }   
                    }  
            }
        }
        System.out.println("SAVING");                
            try {
                    cnn_ae.save( new File("CNN909.zip"), true);
                } catch(Exception e) {
                }
    }
    
    private static void test() throws Exception{

        INDArray[] INPs = new INDArray[1];
        int offsetX = 0;
        int offsetY = 0;
        int hemanth = 0;
        int zach = 0;
        
        System.out.println(" Benchmark: 0.50");
        
        for(int i=0; i<20;i++){
        INDArray img = Img2INDArray.load_image("Images/Test/0/"+i+".jpg",100,100,offsetX,offsetY,0, true);
       
   
//        Img2INDArray.preview_image.setSize(600,600);
    
        
        double best_prob = -1.0;
        //double best_prob2 = -1.0;
        String banner = "####################################################";
        //for (offsetX=0; offsetX<550; offsetX+=10)
       //{
            Img2INDArray.preview_image.setTitle(banner);
            //for (offsetY=0; offsetY<550; offsetY+=10)
            //{
                /** Grab the pixel information from the image at location offsetX,OffsetY of width x height of 94x128 pixels */
               // INDArray subImg1 = img.get( NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(offsetX,offsetX+100), NDArrayIndex.interval(offsetY,offsetY+100) );
                
                
                INPs[0] = img;
                
           
                
                INDArray[] res = cnn_ae.output( INPs );
           
            System.out.println("Unknown"+i+"==>"+ "[["+String.format("%.4f", res[0].getDouble(0))+", "+ String.format("%.4f",res[0].getDouble(1))+"]]");
            
            if (res[0].getDouble(0) > 0.50){
                System.out.println("Found Hemanth!! Granted Access");
                System.out.println();
                hemanth++;
            }
            
            else if (res[0].getDouble(1) > 0.50){
                System.out.println("Found Zach!! Granted Access");
                System.out.println();
                zach++;
            }
            else{
                System.out.println("User not found!!");
                System.out.println();
            }
            //}

            //banner = banner.substring(0, banner.length()-1 );
       //}
         Img2INDArray.preview_image.dispose();
       
       /** This demo takes a lot of memory in forcing some kind of garbage collection seems to help (eliminated out-of-memory errors) */
       java.lang.Runtime.getRuntime().gc();
    }
        System.out.println("Found Hemanth "+ hemanth+" times");
        System.out.println("Found Zach "+ zach +" times");

    }
    
    private static ComputationGraph nn_init()
        {
            // Some hyperparameters (variable) declarations here.
        double learningRate = 0.0001;   // Learning rate

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)      
            .activation(Activation.ELU)         
            .updater(new Adam(learningRate))    
//            .l2(0.0001)                         // For overfitting
            .graphBuilder()
                
                /* Begin. Start creating the neural network structure (Layers) here */
                
            .addInputs("vector_in")             // Name of the layer(s) for the inputs

            .addLayer("INPUT_I1", new ConvolutionLayer.Builder()  // This is the convolutional layer
                .kernelSize(2,2)                // This is the receptive field.
                .stride(1,1)                    // This should be 1,1.
                .nIn(1)                         // Number of channels in the image. Usually 1 (RGB)     
                .nOut(8)                       // How many feature maps to create
                .build(), "vector_in")                      

           .addLayer("POOL1", new SubsamplingLayer.Builder(PoolingType.MAX) // A pooling layer follows after a convolution layer
                .kernelSize(2,2)                // Size of the pool window. 2x2 is typical
                .stride(2,2)                    // The stride should be the same as the kernel size. 2x2
                .build(), "INPUT_I1")
                    
                /** Add additional convolution + pooling layers here */
                
            .addLayer("CONV1", new ConvolutionLayer.Builder()  
                .kernelSize(2,2)
                .stride(1,1)
                .nIn(8)                                     
                .nOut(16)                                    
                .build(), "POOL1")     
                
            .addLayer("POOL2", new SubsamplingLayer.Builder(PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)   
                .build(), "CONV1")

            /** eventually you change from convolution layers to standard feed forward layers
             * Note: The transition from Convolutional+Pool to Feed Forward does NOT require a nIn().
             * DeepLearning4j computes the correct nIn() for this transition step.
             */    
           .addLayer("FF_0", new DenseLayer.Builder()
                //.nIn(1)
                .nOut(7500) // 28X28
                .build(), "POOL2")
                
           .addLayer("FF_1", new DenseLayer.Builder()
                .nIn(7500)
                .nOut(3500) // 28X28
                .build(), "FF_0")
                
           
            .addLayer("FF_2", new DenseLayer.Builder()
                .nIn(3500)       // nIn() needed from here just like a regular NN
                .nOut(1750)
                .build(), "FF_1")

             .addLayer("FF_3", new DenseLayer.Builder()
                .nIn(1750)       // nIn() needed from here just like a regular NN
                .nOut(870)
                .build(), "FF_2")
             .addLayer("FF_4", new DenseLayer.Builder()
                .nIn(870)       // nIn() needed from here just like a regular NN
                .nOut(420)
                .build(), "FF_3") 
             .addLayer("FF_5", new DenseLayer.Builder()
                .nIn(420)       // nIn() needed from here just like a regular NN
                .nOut(210)
                .build(), "FF_4")
            .addLayer("FF_6", new DenseLayer.Builder()
                .nIn(210)       // nIn() needed from here just like a regular NN
                .nOut(120)
                .build(), "FF_5")
                
                
            .addLayer("FF_7", new DenseLayer.Builder()
                .nIn(120)       // nIn() needed from here just like a regular NN
                .nOut(60)
                .build(), "FF_6")

           .addLayer("OUTPUT_O1", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)  
                .activation( Activation.SOFTMAX )                                         
                .nIn( 60 )                                                                
                .nOut( 10 )            // 0-9                                             
                .build(), "FF_7")

            .setOutputs("OUTPUT_O1")
                
                /** height,width,channels 
                 * Below states the images are of height x width of 128 x 128 with 3 channels (RGB)
                 */
            .setInputTypes(InputType.convolutional(100, 100, 1))    
                
                
                
            .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();


         return net;
        }
}
 /* The Autoencoder Part */   
    /*private static ComputationGraph nn_init2()
        {
            // Some hyperparameters (variable) declarations here.
        double learningRate = 0.0008;   // Learning rate

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)      
            .activation(Activation.ELU)         
            .updater(new Adam(learningRate))    
            //.l2(0.00001)                      
            .graphBuilder()
                
                // Begin. Start creating the neural network structure (Layers) here /
                
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
                            
            
                // The autoencoder part begins. Compress down to 25 nodes 
                
           .addLayer("ENCODER_H1", new DenseLayer.Builder()
                //.nIn(1)       // 18432  // 7743
                .nOut(3500)
                .build(), "SUB3")
                
            .addLayer("ENCODER_H2", new DenseLayer.Builder()
                .nIn(3500)
                .nOut(2500)
                .build(), "ENCODER_H1")

            .addLayer("ENCODER_H3", new DenseLayer.Builder()
                .nIn(2500)
                .nOut(1250)
                .build(), "ENCODER_H2")
            
            .addLayer("ENCODER_H4", new DenseLayer.Builder()
                .nIn(1250)
                .nOut(625)
                .build(), "ENCODER_H3")
        
                // This is the embedding layer /
                
            .addLayer("ENCODER_H5", new DenseLayer.Builder()
                .nIn(625)
                .nOut(300)
                .build(), "ENCODER_H4")
                
            .addLayer("ENCODER_H6", new DenseLayer.Builder()
                .nIn(300)
                .nOut(150)
                .build(), "ENCODER_H5")
                
            .addLayer("ENCODER_H7", new DenseLayer.Builder()
                .nIn(150)
                .nOut(25)
                .build(), "ENCODER_H6")
                
            .addLayer("DECODER_H1", new DenseLayer.Builder()
                .nIn(25)
                .nOut(150)
                .build(), "ENCODER_H7")
                
            .addLayer("DECODER_H2", new DenseLayer.Builder()
                .nIn(150)
                .nOut(300)
                .build(), "DECODER_H1")
            .addLayer("DECODER_H3", new DenseLayer.Builder()
                .nIn(300)
                .nOut(625)
                .build(), "DECODER_H2") 
            .addLayer("DECODER_H4", new DenseLayer.Builder()
                .nIn(625)
                .nOut(1250)
                .build(), "DECODER_H3")
            .addLayer("DECODER_H5", new DenseLayer.Builder()
                .nIn(1250)
                .nOut(2500)
                .build(), "DECODER_H4")
            .addLayer("DECODER_H6", new DenseLayer.Builder()
                .nIn(2500)
                .nOut(3500)
                .build(), "DECODER_H5")
                
                
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
*/












































