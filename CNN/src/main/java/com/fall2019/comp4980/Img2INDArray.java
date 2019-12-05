/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.fall2019.comp4980;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * @author John
 */
public class Img2INDArray {

    public static PreviewImage preview_image;
    private static int image_type = BufferedImage.TYPE_INT_RGB;
    private static int channels = 3;
    private static double normalizer = 255.0;

    public static void grayscale() {
        image_type = BufferedImage.TYPE_BYTE_GRAY;
        channels = 1;
        normalizer = 255.0;
    }


    public static void bw() {
        image_type = BufferedImage.TYPE_BYTE_BINARY;
        channels = 1;
        normalizer = 1.0;
    }


    public static void rgb() {
        image_type = BufferedImage.TYPE_INT_RGB;
        channels = 3;
        normalizer = 255.0;
    }

    public static INDArray load_image(String filename, int width, int height, int offsetX, int offsetY, double level, boolean preview) throws Exception {

        BufferedImage img = ImageIO.read(new File(filename));

        return preProcess(width, height, img, offsetX, offsetY, level, preview);

    }

    public static INDArray preProcess(int width, int height, BufferedImage img, int offsetX, int offsetY, double level, boolean preview) {
        corrupt(img, level);

        BufferedImage bi = new BufferedImage(width, height, image_type);


        Graphics bg = bi.getGraphics();
        //bg.drawImage(img, offsetX, offsetY, width,height, null);   
        bg.drawImage(img, offsetX, offsetY, null);
        bg.dispose();

        if (preview) {
            preview_image = new PreviewImage(bi);
            preview_image.setVisible(true);
        }

        return convertImgToINDArray(bi);
    }

    public static void corrupt(BufferedImage img, double level) {
        for (int x = 0; x < img.getWidth(); x++)
            for (int y = 0; y < img.getHeight(); y++)
                if (Math.random() < level) img.setRGB(x, y, (int) (16777216.0 * Math.random()));
    }

    public static INDArray convertImgToINDArray(BufferedImage img) {
        INDArray v = Nd4j.zeros(1, channels, img.getHeight(), img.getWidth());

        getImageBytes(v, img);

        return v;
    }

    public static void getImageBytes(INDArray v, BufferedImage bi) {
        int[] pixel;

        for (int y = 0; y < bi.getWidth(); y++) {
            for (int x = 0; x < bi.getHeight(); x++) {
                pixel = bi.getRaster().getPixel(y, x, new int[channels]);

                for (int c = 0; c < channels; c++)
                    v.putScalar(0, c, x, y, Double.valueOf(pixel[c]) / normalizer);
            }
        }

    }

}







